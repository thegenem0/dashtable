//! Bucket with inline key-value storage.
//!
//! - 12 slots per bucket (configurable)
//! - Fingerprints stored contiguously for SIMD
//! - Keys and values stored inline (no indirection)
//! - Packed control word for slot state
//! - Stash hints for fast stash lookups

use std::{borrow::Borrow, mem::MaybeUninit};

// Number of slots per bucket
pub const SLOTS_PER_BUCKET: usize = 12;

/// Number of stash hint slots per bucket
pub const STASH_HINT_SLOTS: usize = 4;

/// Bucket control word layout (32 bits)
/// - Bits 0-3: count (number of occupied slots)
/// - Bits 4-17: probing bitmap (slot is displaced/not in home bucket)
/// - Bits 18-31: busy bitmap (slots occupied)
#[derive(Clone, Copy, Default)]
struct BucketCtrl(u32);

#[allow(dead_code)]
impl BucketCtrl {
    const COUNT_MASK: u32 = 0xF; // bits 0-3
    const PROBE_SHIFT: u32 = 4;
    const PROBE_MASK: u32 = 0xFFF << Self::PROBE_SHIFT; // bits 4-17 (14 bits)
    const BUSY_SHIFT: u32 = 18;
    const BUSY_MASK: u32 = 0x3FFF << Self::BUSY_SHIFT; // bits 18-31 (14 bits)

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn count(&self) -> u8 {
        (self.0 & Self::COUNT_MASK) as u8
    }

    #[inline]
    pub const fn busy(&self) -> u16 {
        ((self.0 >> Self::BUSY_SHIFT) & 0x3FFF) as u16
    }

    #[inline]
    pub const fn probing(&self) -> u16 {
        ((self.0 >> Self::PROBE_SHIFT) & 0x3FFF) as u16
    }

    #[inline]
    pub const fn is_busy(&self, slot: usize) -> bool {
        (self.0 & (1 << (slot + Self::BUSY_SHIFT as usize))) != 0
    }

    #[inline]
    pub const fn is_probing(&self, slot: usize) -> bool {
        (self.0 & (1 << (slot + Self::PROBE_SHIFT as usize))) != 0
    }

    #[inline]
    pub fn set_busy(&mut self, slot: usize, displaced: bool) {
        debug_assert!(slot < SLOTS_PER_BUCKET);
        // set busy bit
        self.0 |= 1 << (slot + Self::BUSY_SHIFT as usize);

        if displaced {
            // set probe bit
            self.0 |= 1 << (slot + Self::PROBE_SHIFT as usize);
        }

        // increment count (low 4 bits)
        let count = self.count();
        self.0 = (self.0 & !Self::COUNT_MASK) | ((count + 1) as u32);
    }

    #[inline]
    pub fn clear(&mut self, slot: usize) {
        debug_assert!(slot < SLOTS_PER_BUCKET);
        let was_busy = self.is_busy(slot);

        // clear busy bit
        self.0 &= !(1 << (slot + Self::BUSY_SHIFT as usize));

        // clear probe bit
        self.0 &= !(1 << (slot + Self::PROBE_SHIFT as usize));

        if was_busy {
            let count = self.count();
            self.0 = (self.0 & !Self::COUNT_MASK) | ((count - 1) as u32);
        }
    }
}

/// Stash hints for tracking entries that overflow to stash buckets
///
/// Each bucket can track up to 4 stash entries with their fingerprints
/// and which stash bucket they reside in. This allows fingerprint-first
/// rejection even for stash lookups.
///
/// Layout:
/// - `busy`: bits 0-3 indicate which hitnt slots are used, bit 4 is "has stash" flag
/// - `positions`: 2 bits per slot encoding stash bucket index (0-3)
/// - `probe_mask`: bits 0-3 indicate if entry is from neighbour (1) or owner (0)
/// - `fingerprints`: the fingerprints of stash entries
#[derive(Clone, Copy)]
pub struct StashHints {
    /// Bit 0-3: slot occupancy, Bit 4: stash present flag
    busy: u8,

    /// 2 bits per slot: which stash bucket (0-3)
    /// slot 0 = bits 0-1, slot 1 = bits 2-3, etc.
    positions: u8,

    /// Bit i: if set, hint slot i belongs to neighbour bucket
    probe_mask: u8,

    /// Fingerprints of stash entris (4 slots)
    fingerprints: [u8; STASH_HINT_SLOTS],
}

impl Default for StashHints {
    fn default() -> Self {
        Self::new()
    }
}

impl StashHints {
    const STASH_PRESENT_BIT: u8 = 1 << 4;
    const SLOT_MASK: u8 = 0x0F; // bits 0-3

    #[inline]
    pub const fn new() -> Self {
        Self {
            busy: 0,
            positions: 0,
            probe_mask: 0,
            fingerprints: [0; STASH_HINT_SLOTS],
        }
    }

    /// Check if any stash entries exist for this bucket
    #[inline]
    pub const fn has_stash(&self) -> bool {
        (self.busy & Self::STASH_PRESENT_BIT) != 0
    }

    /// Check if hint slot is occupied
    #[inline]
    pub const fn is_slot_busy(&self, slot: usize) -> bool {
        (self.busy & (1 << slot)) != 0
    }

    /// Get stash bucket index for hint slot (0-3)
    #[inline]
    pub const fn get_stash_bucket(&self, slot: usize) -> usize {
        ((self.positions >> (slot * 2)) & 0x3) as usize
    }

    /// Check if hint slot entry belongs to neighbour bucket
    #[inline]
    pub const fn is_from_neighbour(&self, slot: usize) -> bool {
        (self.probe_mask & (1 << slot)) != 0
    }

    /// Find empty hint slot
    #[inline]
    pub fn find_empty_slot(&self) -> Option<usize> {
        let busy = self.busy & Self::SLOT_MASK;
        if busy == Self::SLOT_MASK {
            None // all slots occupied
        } else {
            Some((!busy & Self::SLOT_MASK).trailing_zeros() as usize)
        }
    }

    /// Add stash hint entry
    /// Return true if added, false if all slots are occupied
    pub fn add(&mut self, fp: u8, stash_bucket: usize, from_neighbour: bool) -> bool {
        debug_assert!(stash_bucket < STASH_HINT_SLOTS);

        if let Some(slot) = self.find_empty_slot() {
            self.fingerprints[slot] = fp;

            // set position bits (2)
            let pos_shift = slot * 2;
            self.positions =
                (self.positions & !(0x3 << pos_shift)) | ((stash_bucket as u8) << pos_shift);

            // set probe mask
            if from_neighbour {
                self.probe_mask |= 1 << slot;
            } else {
                self.probe_mask &= !(1 << slot);
            }

            self.busy |= (1 << slot) | Self::STASH_PRESENT_BIT;

            true
        } else {
            false
        }
    }

    /// Remove stash hint entry by fingerprint and stash bucket
    /// Return true if found and removed
    pub fn remove(&mut self, fp: u8, stash_bucket: usize) -> bool {
        for slot in 0..STASH_HINT_SLOTS {
            if self.is_slot_busy(slot)
                && self.fingerprints[slot] == fp
                && self.get_stash_bucket(slot) == stash_bucket
            {
                // clear slot
                self.busy &= !(1 << slot);
                self.fingerprints[slot] = 0;

                // clear stash presence flag if no more hints
                if (self.busy & Self::SLOT_MASK) == 0 {
                    self.busy &= !Self::STASH_PRESENT_BIT;
                }

                return true;
            }
        }
        false
    }

    // Find stash buckets that might contain entries matching fingerprint
    // Return iterator over (stash_bucket_idx, from_neighbour) pairs
    pub fn find_matching(&self, fp: u8) -> impl Iterator<Item = (usize, bool)> + '_ {
        (0..STASH_HINT_SLOTS).filter_map(move |slot| {
            if self.is_slot_busy(slot) && self.fingerprints[slot] == fp {
                Some((self.get_stash_bucket(slot), self.is_from_neighbour(slot)))
            } else {
                None
            }
        })
    }

    /// Get all occupied hint slots as (fingerprint, stash_bucket, from_neighbour)
    pub fn iter(&self) -> impl Iterator<Item = (u8, usize, bool)> + '_ {
        (0..STASH_HINT_SLOTS).filter_map(move |slot| {
            if self.is_slot_busy(slot) {
                Some((
                    self.fingerprints[slot],
                    self.get_stash_bucket(slot),
                    self.is_from_neighbour(slot),
                ))
            } else {
                None
            }
        })
    }

    /// Count used hint slots
    #[inline]
    pub fn count(&self) -> usize {
        (self.busy & Self::SLOT_MASK).count_ones() as usize
    }

    /// Chech if hints are full
    #[inline]
    pub fn is_full(&self) -> bool {
        (self.busy & Self::SLOT_MASK) == Self::SLOT_MASK
    }

    /// Clear all hints
    #[inline]
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

/// Bucket with inline key-value storage
///
/// Memory layout (for K=V=u64, 12 slots):
/// ```text
/// fingerprints: [u8; 12]    = 12 bytes
/// stash_hints:  StashHints  = 7 bytes (busy, positions, probe_mask, fps[4])
/// _pad:         u8          = 1 byte
/// ctrl:         u32         = 4 bytes
/// version:      u32         = 4 bytes
/// keys:         [K; 12]     = 96 bytes (for u64)
/// values:       [V; 12]     = 96 bytes (for u64)
/// -------------------------------------------------
/// Total:                    = 220 bytes (for u64 K/V)
/// ```
#[repr(C)]
pub struct Bucket<K, V> {
    /// Fingerprints (8-bit hash prefixes)
    fingerprints: [u8; SLOTS_PER_BUCKET],

    /// Stash hints for tracking overflow entries
    stash_hints: StashHints,

    /// Padding
    _pad: u8,

    /// Control word: count, probe bitmap, busy bitmap
    ctrl: BucketCtrl,

    /// Version counter
    version: u32,

    /// Keys (inline)
    keys: [MaybeUninit<K>; SLOTS_PER_BUCKET],

    /// Values (inline)
    values: [MaybeUninit<V>; SLOTS_PER_BUCKET],
}

impl<K, V> Bucket<K, V> {
    /// Create new empty bucket
    pub fn new() -> Self {
        Self {
            fingerprints: [0; SLOTS_PER_BUCKET],
            stash_hints: StashHints::new(),
            _pad: 0,
            ctrl: BucketCtrl::empty(),
            version: 0,
            // Safety: MaybeUninit does not require initialization
            keys: unsafe { MaybeUninit::uninit().assume_init() },
            values: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    /// Number of occupied slots
    #[inline]
    pub fn count(&self) -> usize {
        self.ctrl.count() as usize
    }

    /// Check if bucket is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count() >= SLOTS_PER_BUCKET
    }

    /// Check if bucket is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Get busy bitmap
    #[inline]
    pub fn busy_mask(&self) -> u16 {
        self.ctrl.busy()
    }

    /// Get probing bitmap (which slots hold displaced entries)
    #[inline]
    pub fn probing_mask(&self) -> u16 {
        self.ctrl.probing()
    }

    /// Get bucket version
    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Chech if bucket has entries in stash
    #[inline]
    pub fn has_stash_entries(&self) -> bool {
        self.stash_hints.has_stash()
    }

    /// Get reference to stash hints
    #[inline]
    pub fn stash_hints(&self) -> &StashHints {
        &self.stash_hints
    }

    /// Get mutable reference to stash hints
    #[inline]
    pub fn stash_hints_mut(&mut self) -> &mut StashHints {
        &mut self.stash_hints
    }

    /// Add a stash hint for an entry that overflowed to stash bucket
    /// Return true if hint was added, false if all slots are occupied
    #[inline]
    pub fn add_stash_hint(&mut self, fp: u8, stash_bucket: usize, from_neighbour: bool) -> bool {
        self.stash_hints.add(fp, stash_bucket, from_neighbour)
    }

    /// Remove a stash hint when entry is removed from stash bucket
    /// Returns true if hint was found and removed
    #[inline]
    pub fn remove_stash_hint(&mut self, fp: u8, stash_bucket: usize) -> bool {
        self.stash_hints.remove(fp, stash_bucket)
    }

    /// Find stash buckets that might contain the given fingerprint
    #[inline]
    pub fn find_stash_hints(&self, fp: u8) -> impl Iterator<Item = (usize, bool)> + '_ {
        self.stash_hints.find_matching(fp)
    }

    /// Check if stash hints are full (would need overflow handling)
    #[inline]
    pub fn stash_hints_full(&self) -> bool {
        self.stash_hints.is_full()
    }

    /// Find slots matching fingerprint (returns bitmask)
    #[inline]
    pub fn find_fingerprint_mask(&self, fp: u8) -> u16 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse2") {
                return unsafe { self.find_fingerprint_mask_sse2(fp) };
            }
        }

        self.find_fingerprint_mask_scalar(fp)
    }

    /// Scalar fingerprint search
    fn find_fingerprint_mask_scalar(&self, fp: u8) -> u16 {
        let mut mask = 0u16;
        let busy = self.ctrl.busy();
        for i in 0..SLOTS_PER_BUCKET {
            if (busy & (1 << i)) != 0 && self.fingerprints[i] == fp {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// SSE2 fingerprint search
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn find_fingerprint_mask_sse2(&self, fp: u8) -> u16 {
        use std::arch::x86_64::*;

        // load fingerprints (only 12 bytes, but load 16 for simd)
        let mut fp_array = [0u8; 16];
        fp_array[..SLOTS_PER_BUCKET].copy_from_slice(&self.fingerprints);
        let fp_vec = _mm_loadu_si128(fp_array.as_ptr() as *const __m128i);

        // broadcast target
        let target = _mm_set1_epi8(fp as i8);

        // compare
        let cmp = _mm_cmpeq_epi8(fp_vec, target);
        let match_mask = _mm_movemask_epi8(cmp) as u16;

        // AND with busy mask and slot mask (only first 12 bits are valid)
        match_mask & self.ctrl.busy() & 0x0FFF
    }

    /// Find first empty slot
    #[inline]
    pub fn find_empty(&self) -> Option<usize> {
        let busy = self.ctrl.busy();
        // invert busy to get emtpy, mask to valid slot range
        let empty = !busy & 0x0FFF;
        if empty != 0 {
            Some(empty.trailing_zeros() as usize)
        } else {
            None
        }
    }

    /// Iterate over occupied slots
    #[inline]
    pub fn iter_occupied(&self) -> impl Iterator<Item = usize> + '_ {
        let busy = self.ctrl.busy();
        (0..SLOTS_PER_BUCKET).filter(move |&i| (busy & (1 << i)) != 0)
    }

    /// Clear all slots in bucket
    pub fn clear(&mut self) {
        let busy = self.ctrl.busy();
        for slot in 0..SLOTS_PER_BUCKET {
            if (busy & (1 << slot)) != 0 {
                unsafe {
                    self.keys[slot].assume_init_drop();
                    self.values[slot].assume_init_drop();
                }
            }
        }
        self.fingerprints = [0; SLOTS_PER_BUCKET];
        self.ctrl = BucketCtrl::empty();
        self.version = self.version.wrapping_add(1);
    }
}

impl<K, V> Bucket<K, V>
where
    K: Eq,
{
    /// Find slot containing key (if any)
    pub fn find_key<Q>(&self, fp: u8, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let mut mask = self.find_fingerprint_mask(fp);
        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            // Safety: slot is occupied (from fp match which checks busy)
            let slot_key = unsafe { self.keys[slot].assume_init_ref() };
            if slot_key.borrow() == key {
                return Some(slot);
            }
            mask &= mask - 1; // clear lowest bit
        }
        None
    }

    /// Get reference to key at slot
    ///
    /// # Safety
    /// Caller must ensure slot is occupied
    #[inline]
    pub unsafe fn get_key(&self, slot: usize) -> &K {
        debug_assert!(self.ctrl.is_busy(slot));
        self.keys[slot].assume_init_ref()
    }

    /// Get reference to value at slot
    ///
    /// # Safety
    /// Caller must ensure slot is occupied
    #[inline]
    pub unsafe fn get_value(&self, slot: usize) -> &V {
        debug_assert!(self.ctrl.is_busy(slot));
        self.values[slot].assume_init_ref()
    }

    /// Get mutable reference to value at slot
    ///
    /// # Safety
    /// Caller must ensure slot is occupied
    #[inline]
    pub unsafe fn get_value_mut(&mut self, slot: usize) -> &mut V {
        debug_assert!(self.ctrl.is_busy(slot));
        self.values[slot].assume_init_mut()
    }

    /// Get fingerprint at slot
    #[inline]
    pub fn get_fingerprint(&self, slot: usize) -> u8 {
        self.fingerprints[slot]
    }

    /// Check if slot consaints a displaced entry (not in home bucket)
    #[inline]
    pub fn is_displaced(&self, slot: usize) -> bool {
        self.ctrl.is_probing(slot)
    }

    /// Insert key-value pair at slot
    ///
    /// # Panics
    /// Debug panics if slot is occupied
    pub fn insert_at(&mut self, slot: usize, fp: u8, key: K, value: V, displaced: bool) {
        debug_assert!(!self.ctrl.is_busy(slot), "slot already occupied");
        debug_assert!(slot < SLOTS_PER_BUCKET);

        self.fingerprints[slot] = fp;
        self.keys[slot].write(key);
        self.values[slot].write(value);
        self.ctrl.set_busy(slot, displaced);
        self.version = self.version.wrapping_add(1);
    }

    /// Remove entry at slot, returning key and value
    ///
    /// # Panics
    /// Debug panics if slot is not occupied
    pub fn remove_at(&mut self, slot: usize) -> (K, V) {
        debug_assert!(self.ctrl.is_busy(slot), "slot not occupied");

        self.fingerprints[slot] = 0;
        // Safety: slot is occupied
        let key = unsafe { self.keys[slot].assume_init_read() };
        let value = unsafe { self.values[slot].assume_init_read() };
        self.ctrl.clear(slot);
        self.version = self.version.wrapping_add(1);

        (key, value)
    }

    /// Try to insert, returns slot index if successful
    pub fn try_insert(&mut self, fp: u8, key: K, value: V, displaced: bool) -> Option<usize> {
        let slot = self.find_empty()?;
        self.insert_at(slot, fp, key, value, displaced);
        Some(slot)
    }
}

impl<K, V> Default for Bucket<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for Bucket<K, V> {
    fn drop(&mut self) {
        // drop all occupied entried
        let busy = self.ctrl.busy();
        for slot in 0..SLOTS_PER_BUCKET {
            if (busy & (1 << slot)) != 0 {
                unsafe {
                    self.keys[slot].assume_init_drop();
                    self.values[slot].assume_init_drop();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stash_hints_new() {
        let hints = StashHints::new();
        assert!(!hints.has_stash());
        assert_eq!(hints.count(), 0);
        assert!(!hints.is_full());
    }

    #[test]
    fn test_stash_hints_add_remove() {
        let mut hints = StashHints::new();

        // Add first hint
        assert!(hints.add(0xAA, 2, false));
        assert!(hints.has_stash());
        assert_eq!(hints.count(), 1);
        assert!(hints.is_slot_busy(0));
        assert_eq!(hints.get_stash_bucket(0), 2);
        assert!(!hints.is_from_neighbour(0));

        // Add second hint from neighbor
        assert!(hints.add(0xBB, 1, true));
        assert_eq!(hints.count(), 2);
        assert!(hints.is_from_neighbour(1));

        // Remove first hint
        assert!(hints.remove(0xAA, 2));
        assert_eq!(hints.count(), 1);
        assert!(!hints.is_slot_busy(0));
        assert!(hints.has_stash()); // Still has one hint

        // Remove second hint
        assert!(hints.remove(0xBB, 1));
        assert_eq!(hints.count(), 0);
        assert!(!hints.has_stash());
    }

    #[test]
    fn test_stash_hints_full() {
        let mut hints = StashHints::new();

        // Fill all 4 slots
        for i in 0..4 {
            assert!(hints.add(i as u8, i, false));
        }
        assert!(hints.is_full());
        assert_eq!(hints.count(), 4);

        // Should fail to add more
        assert!(!hints.add(0xFF, 0, false));
    }

    #[test]
    fn test_stash_hints_find_matching() {
        let mut hints = StashHints::new();

        hints.add(0xAA, 0, false);
        hints.add(0xBB, 1, true);
        hints.add(0xAA, 2, false); // Same fingerprint, different stash bucket

        let matches: Vec<_> = hints.find_matching(0xAA).collect();
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, false)));
        assert!(matches.contains(&(2, false)));

        let matches: Vec<_> = hints.find_matching(0xBB).collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (1, true));

        let matches: Vec<_> = hints.find_matching(0xCC).collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_bucket_ctrl_layout() {
        let mut ctrl = BucketCtrl::empty();
        assert_eq!(ctrl.count(), 0);
        assert_eq!(ctrl.busy(), 0);
        assert_eq!(ctrl.probing(), 0);

        // Set slot 0 as busy (not displaced)
        ctrl.set_busy(0, false);
        assert_eq!(ctrl.count(), 1);
        assert!(ctrl.is_busy(0));
        assert!(!ctrl.is_probing(0));

        // Set slot 5 as busy and displaced
        ctrl.set_busy(5, true);
        assert_eq!(ctrl.count(), 2);
        assert!(ctrl.is_busy(5));
        assert!(ctrl.is_probing(5));

        // Clear slot 0
        ctrl.clear(0);
        assert_eq!(ctrl.count(), 1);
        assert!(!ctrl.is_busy(0));
    }

    #[test]
    fn test_empty_bucket() {
        let bucket: Bucket<String, i32> = Bucket::new();
        assert!(bucket.is_empty());
        assert!(!bucket.is_full());
        assert_eq!(bucket.count(), 0);
        assert_eq!(bucket.busy_mask(), 0);
        assert_eq!(bucket.probing_mask(), 0);
    }

    #[test]
    fn test_insert_and_find() {
        let mut bucket: Bucket<String, i32> = Bucket::new();

        bucket.insert_at(0, 0xAB, "hello".to_string(), 42, false);

        assert_eq!(bucket.count(), 1);
        assert!(!bucket.is_empty());
        assert!(!bucket.is_displaced(0));

        let slot = bucket.find_key(0xAB, &"hello".to_string());
        assert_eq!(slot, Some(0));

        let value = unsafe { bucket.get_value(0) };
        assert_eq!(*value, 42);
    }

    #[test]
    fn test_displaced_insert() {
        let mut bucket: Bucket<u32, u32> = Bucket::new();

        bucket.insert_at(0, 0xAA, 1, 10, false);
        bucket.insert_at(1, 0xBB, 2, 20, true); // displaced

        assert!(!bucket.is_displaced(0));
        assert!(bucket.is_displaced(1));
        assert_eq!(bucket.probing_mask() & (1 << 1), 1 << 1);
    }

    #[test]
    fn test_remove() {
        let mut bucket: Bucket<String, i32> = Bucket::new();
        bucket.insert_at(5, 0xCD, "test".to_string(), 99, false);

        let (key, value) = bucket.remove_at(5);
        assert_eq!(key, "test");
        assert_eq!(value, 99);
        assert!(bucket.is_empty());
    }

    #[test]
    fn test_fingerprint_collision() {
        let mut bucket: Bucket<u64, u64> = Bucket::new();

        // Insert multiple with same fingerprint
        bucket.insert_at(0, 0xAA, 100, 1, false);
        bucket.insert_at(3, 0xAA, 200, 2, false);
        bucket.insert_at(7, 0xAA, 300, 3, false);

        // Find correct one by key
        assert_eq!(bucket.find_key(0xAA, &100), Some(0));
        assert_eq!(bucket.find_key(0xAA, &200), Some(3));
        assert_eq!(bucket.find_key(0xAA, &300), Some(7));
        assert_eq!(bucket.find_key(0xAA, &999), None);
    }

    #[test]
    fn test_full_bucket() {
        let mut bucket: Bucket<u32, u32> = Bucket::new();

        for i in 0..SLOTS_PER_BUCKET {
            assert!(bucket
                .try_insert(i as u8, i as u32, i as u32 * 10, false)
                .is_some());
        }

        assert!(bucket.is_full());
        assert!(bucket.try_insert(0xFF, 999, 999, false).is_none());
    }

    #[test]
    fn test_find_empty() {
        let mut bucket: Bucket<u32, u32> = Bucket::new();
        assert_eq!(bucket.find_empty(), Some(0));

        bucket.insert_at(0, 0x11, 1, 10, false);
        assert_eq!(bucket.find_empty(), Some(1));

        bucket.insert_at(1, 0x22, 2, 20, false);
        assert_eq!(bucket.find_empty(), Some(2));
    }

    #[test]
    fn test_iter_occupied() {
        let mut bucket: Bucket<u32, u32> = Bucket::new();

        bucket.insert_at(1, 0xAA, 10, 100, false);
        bucket.insert_at(5, 0xBB, 50, 500, false);
        bucket.insert_at(9, 0xCC, 90, 900, false);

        let occupied: Vec<_> = bucket.iter_occupied().collect();
        assert_eq!(occupied, vec![1, 5, 9]);
    }

    #[test]
    fn test_version_increments() {
        let mut bucket: Bucket<u32, u32> = Bucket::new();
        let initial_version = bucket.version();

        bucket.insert_at(0, 0xAA, 1, 10, false);
        assert_eq!(bucket.version(), initial_version + 1);

        bucket.remove_at(0);
        assert_eq!(bucket.version(), initial_version + 2);
    }

    #[test]
    fn test_clear() {
        let mut bucket: Bucket<String, i32> = Bucket::new();

        bucket.insert_at(0, 0xAA, "one".to_string(), 1, false);
        bucket.insert_at(1, 0xBB, "two".to_string(), 2, false);
        assert_eq!(bucket.count(), 2);

        bucket.clear();
        assert!(bucket.is_empty());
        assert_eq!(bucket.find_key(0xAA, &"one".to_string()), None);
    }

    #[test]
    fn test_busy_mask() {
        let mut bucket: Bucket<u32, u32> = Bucket::new();

        bucket.insert_at(0, 0x11, 1, 10, false);
        bucket.insert_at(3, 0x22, 2, 20, false);
        bucket.insert_at(7, 0x33, 3, 30, false);

        let mask = bucket.busy_mask();
        assert_eq!(mask, 0b10001001); // bits 0, 3, 7
    }

    #[test]
    fn test_fingerprint_mask_scalar() {
        let mut bucket: Bucket<u32, u32> = Bucket::new();

        bucket.insert_at(0, 0xAA, 1, 10, false);
        bucket.insert_at(2, 0xBB, 2, 20, false);
        bucket.insert_at(4, 0xAA, 3, 30, false);
        bucket.insert_at(6, 0xCC, 4, 40, false);

        let mask = bucket.find_fingerprint_mask(0xAA);
        assert_eq!(mask, 0b00010001); // bits 0, 4
    }
}
