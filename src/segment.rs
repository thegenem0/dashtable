//! Segment is a mini hash table containing 64 regular + 4 stash buckets.
//!
//! - 64 regular buckets for normal entries
//! - 4 stash buckets for overflow
//! - 2-bucket probing for insert: home -> neighbour
//! - 4-bucket probing for find: home → prev → next → next+1
//! - Stash hints for fast stash lookups
//! - Local depth tracking for extendible hashing

use std::borrow::Borrow;

use crate::bucket::{Bucket, SLOTS_PER_BUCKET};

/// Number of regular buckets
pub const REGULAR_BUCKETS: usize = 64;

/// Number of stash buckets (overflow)
pub const STASH_BUCKETS: usize = 4;

/// Total buckets per segment
pub const TOTAL_BUCKETS: usize = REGULAR_BUCKETS + STASH_BUCKETS;

/// Total segment capacity
pub const SEGMENT_CAPACITY: usize = TOTAL_BUCKETS * SLOTS_PER_BUCKET;

/// Number of buckets in the probing sequence
pub const PROBE_COUNT: usize = 4;

/// Segment containing buckets and metadata
pub struct Segment<K, V> {
    /// Regular buckets (idexed 0..63)
    regular: Box<[Bucket<K, V>; REGULAR_BUCKETS]>,

    /// Stash buckets for overflow (indexed 64..67)
    stash: [Bucket<K, V>; STASH_BUCKETS],

    /// Local depth for extendible hashing
    local_depth: u8,

    /// Overflow counter: incremented when stash hints are exhausted
    /// When > 0, must scan all stash buckets on lookup
    overflow_count: u8,

    /// Total entry count in segment
    count: u32,
}

impl<K, V> Segment<K, V> {
    /// Create new segment with provided local depth
    pub fn new(local_depth: u8) -> Self {
        Self {
            regular: Box::new(std::array::from_fn(|_| Bucket::new())),
            stash: std::array::from_fn(|_| Bucket::new()),
            local_depth,
            overflow_count: 0,
            count: 0,
        }
    }

    /// Get local depth
    #[inline]
    pub fn local_depth(&self) -> u8 {
        self.local_depth
    }

    /// Set local depth
    #[inline]
    pub fn set_local_depth(&mut self, depth: u8) {
        self.local_depth = depth;
    }

    /// Get entry count
    #[inline]
    pub fn count(&self) -> usize {
        self.count as usize
    }

    /// Check if segment is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Check if segment is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count() >= SEGMENT_CAPACITY
    }

    /// Get load factor (0.0 to 1.0)
    #[inline]
    pub fn load_factor(&self) -> f64 {
        self.count() as f64 / SEGMENT_CAPACITY as f64
    }

    /// Get overflow count
    #[inline]
    pub fn overflow_count(&self) -> u8 {
        self.overflow_count
    }

    /// Get reference to regular bucket by index (0..63)
    #[inline]
    pub fn bucket(&self, idx: usize) -> &Bucket<K, V> {
        debug_assert!(idx < REGULAR_BUCKETS);
        &self.regular[idx]
    }

    /// Get mutable reference to regular bucket by index (0..63)
    #[inline]
    pub fn bucket_mut(&mut self, idx: usize) -> &mut Bucket<K, V> {
        debug_assert!(idx < REGULAR_BUCKETS);
        &mut self.regular[idx]
    }

    /// Get reference to stash bucket by index (0..3)
    #[inline]
    pub fn stash_bucket(&self, idx: usize) -> &Bucket<K, V> {
        debug_assert!(idx < STASH_BUCKETS);
        &self.stash[idx]
    }

    /// Get mutable reference to stash bucket by index (0..3)
    #[inline]
    pub fn stash_bucket_mut(&mut self, idx: usize) -> &mut Bucket<K, V> {
        debug_assert!(idx < STASH_BUCKETS);
        &mut self.stash[idx]
    }

    /// Compute home bucket index from hash
    #[inline]
    pub fn home_bucket(hash: u64) -> usize {
        (hash as usize) % REGULAR_BUCKETS
    }

    /// Compute pprevious bucket index (wraps 0 -> 63)
    #[inline]
    pub fn prev_bucket(idx: usize) -> usize {
        idx.wrapping_sub(1) % REGULAR_BUCKETS
    }

    /// Compute next bucket index (wraps 63 -> 0)
    #[inline]
    pub fn next_bucket(idx: usize) -> usize {
        (idx + 1) % REGULAR_BUCKETS
    }

    /// Compute 4-bucket probing sequence from hash
    ///
    /// Probing sequence: home → prev → next → next+1
    #[inline]
    pub fn probe_sequence(hash: u64) -> [usize; PROBE_COUNT] {
        let home = Self::home_bucket(hash);
        let prev = Self::prev_bucket(home);
        let next = Self::next_bucket(home);
        let next2 = Self::next_bucket(next);

        [home, prev, next, next2]
    }

    /// Get bucket reference by index (handles regular and stash)
    #[inline]
    pub fn get_bucket(&self, idx: usize) -> &Bucket<K, V> {
        if idx < REGULAR_BUCKETS {
            &self.regular[idx]
        } else {
            &self.stash[idx - REGULAR_BUCKETS]
        }
    }

    /// Get mutable bucket reference by index (handles regular and stash)
    #[inline]
    pub fn get_bucket_mut(&mut self, idx: usize) -> &mut Bucket<K, V> {
        if idx < REGULAR_BUCKETS {
            &mut self.regular[idx]
        } else {
            &mut self.stash[idx - REGULAR_BUCKETS]
        }
    }

    /// Try adding stash hint to home bucket, falling back to neighbour
    /// Returns true if hint was added, false if overflow occurred
    fn add_stash_hint(&mut self, home: usize, fp: u8, stash_bucket: usize) -> bool {
        // try home bucket
        if self.regular[home].add_stash_hint(fp, stash_bucket, false) {
            return true;
        }

        // try neighbour bucket
        let neighbour = Self::next_bucket(home);
        if self.regular[neighbour].add_stash_hint(fp, stash_bucket, true) {
            return true;
        }

        // both full, increment overflow
        self.overflow_count = self.overflow_count.saturating_add(1);
        false
    }

    /// Remove stash hint from home or neighbour bucket
    fn remove_stash_hint(&mut self, home: usize, fp: u8, stash_bucket: usize) {
        // try home bucket
        if self.regular[home].remove_stash_hint(fp, stash_bucket) {
            return;
        }

        // try neighbour bucket
        let neighbour = Self::next_bucket(home);
        if self.regular[neighbour].remove_stash_hint(fp, stash_bucket) {
            return;
        }

        // was an overflown entry, decrement overflow
        self.overflow_count = self.overflow_count.saturating_sub(1);
    }
}

impl<K, V> Segment<K, V>
where
    K: Eq,
{
    /// Find entry in stash via hints
    /// Returns (stash_bucket_idx, slot_idx) if found
    fn find_in_stash<Q>(&self, home: usize, fp: u8, key: &Q) -> Option<(usize, usize)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let neighbour = Self::next_bucket(home);

        // collect stash buckets to check hints from
        let mut checked = [false; STASH_BUCKETS];

        // check from home bucket
        for (stash_idx, _from_neighbour) in self.regular[home].find_stash_hints(fp) {
            if !checked[stash_idx] {
                checked[stash_idx] = true;
                if let Some(slot) = self.stash[stash_idx].find_key(fp, key) {
                    return Some((REGULAR_BUCKETS + stash_idx, slot));
                }
            }
        }

        // chech from neighbour bucket
        for (stash_idx, _from_neighbour) in self.regular[neighbour].find_stash_hints(fp) {
            if !checked[stash_idx] {
                checked[stash_idx] = true;
                if let Some(slot) = self.stash[stash_idx].find_key(fp, key) {
                    return Some((REGULAR_BUCKETS + stash_idx, slot));
                }
            }
        }

        // if overflow, scan remaining stash buckets
        if self.overflow_count > 0 {
            for (i, bucket) in self.stash.iter().enumerate() {
                if !checked[i] {
                    if let Some(slot) = bucket.find_key(fp, key) {
                        return Some((REGULAR_BUCKETS + i, slot));
                    }
                }
            }
        }

        None
    }

    /// Find entry by hash, fingerprint and key
    ///
    /// Returns (bucket_idx, slot_idx) if found
    /// bucket_idx >= REGULAR_BUCKETS indicates entry is in stash
    pub fn find<Q>(&self, fp: u8, hash: u64, key: &Q) -> Option<(usize, usize)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let probes = Self::probe_sequence(hash);
        let home = probes[0];

        // check regular buckets in probe order
        for &bucket_idx in &probes {
            if let Some(slot) = self.regular[bucket_idx].find_key(fp, key) {
                return Some((bucket_idx, slot));
            }
        }

        // check stash via hints (or full scan if overflow)
        if self.regular[home].has_stash_entries()
            || self.regular[Self::next_bucket(home)].has_stash_entries()
            || self.overflow_count > 0
        {
            return self.find_in_stash(home, fp, key);
        }

        None
    }

    /// Check if key exists
    #[inline]
    pub fn contains<Q>(&self, fp: u8, hash: u64, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.find(fp, hash, key).is_some()
    }

    /// Get reference to value by fingerprint, hash, and key
    pub fn get<Q>(&self, fp: u8, hash: u64, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let (bucket_idx, slot_idx) = self.find(fp, hash, key)?;
        let bucket = self.get_bucket(bucket_idx);
        Some(unsafe { bucket.get_value(slot_idx) })
    }

    /// Get mutable reference to value by hash, fingerprint and key
    pub fn get_mut<Q>(&mut self, fp: u8, hash: u64, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let (bucket_idx, slot_idx) = self.find(fp, hash, key)?;
        let bucket = self.get_bucket_mut(bucket_idx);
        Some(unsafe { bucket.get_value_mut(slot_idx) })
    }

    /// Get key-value pari references
    pub fn get_kv<Q>(&self, fp: u8, hash: u64, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let (bucket_idx, slot_idx) = self.find(fp, hash, key)?;
        let bucket = self.get_bucket(bucket_idx);
        // Safety: find() only returns occupied slots
        unsafe { Some((bucket.get_key(slot_idx), bucket.get_value(slot_idx))) }
    }

    /// Insert entry
    ///
    /// Returns `Ok((bucket_idx, slot_idx))` on success
    /// Returns `Err((key, value))` if segment is full (needs splitting)
    ///
    /// Note: This does NOT check for duplicates. Caller should check first
    /// if upsert semantics are needed.
    pub fn insert(
        &mut self,
        fp: u8,
        hash: u64,
        key: K,
        value: V,
    ) -> Result<(usize, usize), (K, V)> {
        let home = Self::home_bucket(hash);
        let neighbour = Self::next_bucket(home);

        // try home bucket
        if let Some(slot) = self.regular[home].find_empty() {
            self.regular[home].insert_at(slot, fp, key, value, false);
            self.count += 1;
            return Ok((home, slot));
        }

        // try neighbour bucket
        if let Some(slot) = self.regular[neighbour].find_empty() {
            self.regular[neighbour].insert_at(slot, fp, key, value, true);
            self.count += 1;
            return Ok((neighbour, slot));
        }

        // try stash buckets
        for (i, bucket) in self.stash.iter_mut().enumerate() {
            if let Some(slot) = bucket.find_empty() {
                bucket.insert_at(slot, fp, key, value, true); // always displaced
                self.count += 1;
                self.add_stash_hint(home, fp, i);

                return Ok((REGULAR_BUCKETS + i, slot));
            }
        }

        Err((key, value)) // segment is full
    }

    /// Insert entry, checking for existing key first.
    ///
    /// Returns `InsertResult::Inserted` on success.
    /// Returns `InsertResult::Updated(old_value)` if key existed.
    /// Returns `InsertResult::SegmentFull(key, value)` if segment is full.
    pub fn insert_checked(&mut self, fp: u8, hash: u64, key: K, value: V) -> InsertResult<K, V> {
        // check if key exists
        if let Some((bucket_idx, slot_idx)) = self.find(fp, hash, &key) {
            let bucket = self.get_bucket_mut(bucket_idx);
            // Safety: find() only returns occupied slots
            let old_value = std::mem::replace(unsafe { bucket.get_value_mut(slot_idx) }, value);
            return InsertResult::Updated(old_value);
        }

        // try to insert
        match self.insert(fp, hash, key, value) {
            Ok(_) => InsertResult::Inserted,
            Err((k, v)) => InsertResult::SegmentFull(k, v),
        }
    }

    /// Remove entry by hash, fingerprint and key.
    ///
    /// Returns `(key, value)` if found and removed.
    pub fn remove<Q>(&mut self, fp: u8, hash: u64, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let (bucket_idx, slot_idx) = self.find(fp, hash, key)?;

        let (k, v) = self.get_bucket_mut(bucket_idx).remove_at(slot_idx);
        self.count -= 1;

        // if removed, remove stash hint
        if bucket_idx >= REGULAR_BUCKETS {
            let home = Self::home_bucket(hash);
            let stash_idx = bucket_idx - REGULAR_BUCKETS;
            self.remove_stash_hint(home, fp, stash_idx);
        }

        Some((k, v))
    }

    /// Remove entry and return only the value
    pub fn remove_value<Q>(&mut self, hash: u64, fp: u8, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.remove(fp, hash, key).map(|(_, v)| v)
    }

    /// Iterate over all entries in the segment.
    ///
    /// Yields `(bucket_idx, slot_idx, &K, &V)` for each entry.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &K, &V)> {
        let regular_iter = self
            .regular
            .iter()
            .enumerate()
            .flat_map(|(bucket_idx, bucket)| {
                bucket.iter_occupied().map(move |slot_idx| {
                    // Safety: iter_occupied only yields occupied slots
                    let key = unsafe { bucket.get_key(slot_idx) };
                    let value = unsafe { bucket.get_value(slot_idx) };
                    (bucket_idx, slot_idx, key, value)
                })
            });

        let stash_iter = self.stash.iter().enumerate().flat_map(|(i, bucket)| {
            let bucket_idx = REGULAR_BUCKETS + i;
            bucket.iter_occupied().map(move |slot_idx| {
                // Safety: iter_occupied only yields occupied slots
                let key = unsafe { bucket.get_key(slot_idx) };
                let value = unsafe { bucket.get_value(slot_idx) };
                (bucket_idx, slot_idx, key, value)
            })
        });

        regular_iter.chain(stash_iter)
    }

    // /// Iterate over all key-value pairs with mutable values
    // ///
    // /// Yields `(&K, &mut V)` for each entry.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        let regular_iter = self.regular.iter_mut().flat_map(|bucket| bucket.iter_mut());
        let stash_iter = self.stash.iter_mut().flat_map(|bucket| bucket.iter_mut());
        regular_iter.chain(stash_iter)
    }

    /// clear all entries from segment
    pub fn clear(&mut self) {
        for bucket in self.regular.iter_mut() {
            bucket.clear();
        }
        for bucket in self.stash.iter_mut() {
            bucket.clear();
        }
        self.count = 0;
        self.overflow_count = 0;
    }
}

impl<K, V> Segment<K, V>
where
    K: Eq + std::hash::Hash,
{
    /// Split this segment, creating a new sibling segment
    ///
    /// Entries are redistributed based on the next bit of their hash
    /// This segment keeps entries where `(hash >> (64 - new_depth)) & 1 == 0`
    /// The new segment gets entries where `(hash >> (64 - new_depth)) & 1 == 1`
    ///
    /// Both segments will have `local_depth = self.local_depth + 1`.
    ///
    /// `hash_fn` is called to recompute hashes for redistribution.
    pub fn split<F>(&mut self, mut hash_fn: F) -> Segment<K, V>
    where
        F: FnMut(&K) -> (u64, u8), // returns (hash, fingerprint)
    {
        let new_depth = self.local_depth + 1;
        let mut sibling = Segment::new(new_depth);

        // bit position to check (counting from MSB)
        let bit_shift = 64 - new_depth as u32;

        // collect entries to move (can't modify while iterating)
        let mut to_move: Vec<(u64, u8, K, V)> = Vec::new();

        // collect info about stash entries that will remain (for hint rebuild)
        // (stash_bucket_idx, hash, fingerprint)
        let mut remaining_stash: Vec<(usize, u64, u8)> = Vec::new();

        // check regular buckets
        for bucket in self.regular.iter_mut() {
            let mut slot = 0;
            while slot < SLOTS_PER_BUCKET {
                if bucket.busy_mask() & (1 << slot) != 0 {
                    // Safety: slot is occupied
                    let key = unsafe { bucket.get_key(slot) };
                    let (hash, fp) = hash_fn(key);

                    // check discriminating bit
                    if (hash >> bit_shift) & 1 == 1 {
                        // move to sibling
                        let (k, v) = bucket.remove_at(slot);
                        to_move.push((hash, fp, k, v));
                        self.count -= 1;
                        // don't increment slot, next entry is shifted into this position
                        continue;
                    }
                }
                slot += 1;
            }
        }

        // check stash buckets
        for (stash_idx, bucket) in self.stash.iter_mut().enumerate() {
            let mut slot = 0;
            while slot < SLOTS_PER_BUCKET {
                if bucket.busy_mask() & (1 << slot) != 0 {
                    // Safety: slot is occupied
                    let key = unsafe { bucket.get_key(slot) };
                    let (hash, fp) = hash_fn(key);

                    if (hash >> bit_shift) & 1 == 1 {
                        // move to sibling
                        let (k, v) = bucket.remove_at(slot);
                        to_move.push((hash, fp, k, v));
                        self.count -= 1;

                        // don't increment slot
                        continue;
                    } else {
                        remaining_stash.push((stash_idx, hash, fp));
                    }
                }
                slot += 1;
            }
        }

        // clear stash hints from regular buckets only
        for bucket in self.regular.iter_mut() {
            bucket.stash_hints_mut().clear();
        }
        self.overflow_count = 0;

        // rebuild stash hints for remaining stash entries
        for (stash_idx, hash, fp) in remaining_stash {
            let home = Self::home_bucket(hash);
            // add_stash_hint handles overflow internally
            self.add_stash_hint(home, fp, stash_idx);
        }

        // insert moved entries into sibling
        for (hash, fp, key, value) in to_move {
            let result = sibling.insert(fp, hash, key, value);
            debug_assert!(result.is_ok(), "sibling segment full during split");
        }

        // update depth
        self.local_depth = new_depth;

        sibling
    }
}

/// Result of checked insert operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertResult<K, V> {
    /// Entry inserted successfully
    Inserted,

    /// Entry existed, old value returned
    Updated(V),

    /// Segment is full, entry could not be inserted
    /// Key and value were consumed, caller should split and retry
    SegmentFull(K, V),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_segment() {
        let seg: Segment<u64, u64> = Segment::new(1);
        assert!(seg.is_empty());
        assert!(!seg.is_full());
        assert_eq!(seg.local_depth(), 1);
        assert_eq!(seg.overflow_count(), 0);
    }

    #[test]
    fn test_probe_sequence() {
        // Probe sequence is for FIND, not insert
        // Order: home -> prev -> next -> next+1
        let probes = Segment::<u64, u64>::probe_sequence(0);
        assert_eq!(probes[0], 0); // home
        assert_eq!(probes[1], 63); // prev (wraps)
        assert_eq!(probes[2], 1); // next
        assert_eq!(probes[3], 2); // next+1
    }

    #[test]
    fn test_insert_and_find() {
        let mut seg: Segment<u64, u64> = Segment::new(1);

        let result = seg.insert(0xAB, 0x1234, 100, 1000);
        assert!(result.is_ok());
        assert_eq!(seg.count(), 1);

        let value = seg.get(0xAB, 0x1234, &100);
        assert_eq!(value, Some(&1000));
    }

    #[test]
    fn test_stash_with_hints() {
        let mut seg: Segment<u32, u32> = Segment::new(1);

        // Fill home and neighbor buckets for hash 0
        // Home = bucket 0, Neighbor = bucket 1
        // 2-bucket insert: 12 + 12 = 24 slots before stash
        for i in 0..24 {
            let result = seg.insert(i as u8, 0, i, i * 10);
            assert!(result.is_ok(), "Failed to insert {}", i);
        }

        // Next insert should go to stash
        let result = seg.insert(0xAA, 0, 100, 1000);
        assert!(result.is_ok());
        let (bucket_idx, _) = result.unwrap();
        assert!(
            bucket_idx >= REGULAR_BUCKETS,
            "Should be in stash, got {}",
            bucket_idx
        );

        // Should be able to find it via hints
        assert_eq!(seg.get(0xAA, 0, &100), Some(&1000));

        // Home or neighbor bucket should have stash hint
        assert!(seg.bucket(0).has_stash_entries() || seg.bucket(1).has_stash_entries());
    }

    #[test]
    fn test_stash_hint_removal() {
        let mut seg: Segment<u32, u32> = Segment::new(1);

        // Fill home and neighbor (24 slots)
        for i in 0..24 {
            let _ = seg.insert(i as u8, 0, i, i * 10);
        }

        // Insert to stash
        let _ = seg.insert(0xAA, 0, 100, 1000);
        assert!(seg.bucket(0).has_stash_entries() || seg.bucket(1).has_stash_entries());

        // Remove the stash entry
        let removed = seg.remove(0xAA, 0, &100);
        assert_eq!(removed, Some((100, 1000)));
    }

    #[test]
    fn test_overflow_count() {
        let mut seg: Segment<u32, u32> = Segment::new(1);

        // Fill home and neighbor (24 slots)
        for i in 0..24 {
            let _ = seg.insert(i as u8, 0, i, 0);
        }

        // Fill stash buckets - each has 12 slots, 4 stash buckets = 48 slots
        // But only 8 hint slots available (4 per bucket for home + neighbor)
        // After 8 stash entries, overflow_count should increment
        for i in 24..80 {
            let result = seg.insert(i as u8, 0, i, 0);
            if result.is_err() {
                break; // Segment full
            }
        }

        // Check that we can still find entries in regular buckets
        for i in 0..24 {
            assert!(seg.contains(i as u8, 0, &i), "Can't find entry {}", i);
        }
    }

    #[test]
    fn test_get_mut() {
        let mut seg: Segment<u64, u64> = Segment::new(1);
        let _ = seg.insert(0xAB, 0x1234, 100, 1000);

        if let Some(value) = seg.get_mut(0xAB, 0x1234, &100) {
            *value = 2000;
        }

        assert_eq!(seg.get(0xAB, 0x1234, &100), Some(&2000));
    }

    #[test]
    fn test_remove() {
        let mut seg: Segment<u64, u64> = Segment::new(1);

        let _ = seg.insert(0xAB, 0x1234, 100, 1000);
        let removed = seg.remove(0xAB, 0x1234, &100);

        assert_eq!(removed, Some((100, 1000)));
        assert!(seg.is_empty());
        assert_eq!(seg.get(0xAB, 0x1234, &100), None);
    }

    #[test]
    fn test_insert_checked_update() {
        let mut seg: Segment<u64, u64> = Segment::new(1);

        // First insert
        let result = seg.insert_checked(0xAB, 0x1234, 100, 1000);
        assert_eq!(result, InsertResult::Inserted);

        // Update existing
        let result = seg.insert_checked(0xAB, 0x1234, 100, 2000);
        assert_eq!(result, InsertResult::Updated(1000));

        // Verify update
        assert_eq!(seg.get(0xAB, 0x1234, &100), Some(&2000));
        assert_eq!(seg.count(), 1);
    }

    #[test]
    fn test_probing_overflow() {
        let mut seg: Segment<u32, u32> = Segment::new(1);

        // Fill home bucket (bucket 0) completely - 12 slots
        for i in 0..SLOTS_PER_BUCKET {
            let result = seg.insert(i as u8, 0, i as u32, i as u32 * 10);
            assert!(result.is_ok());
            let (bucket_idx, _) = result.unwrap();
            assert_eq!(bucket_idx, 0, "should be in home bucket");
        }

        // Next insert should go to neighbor bucket (bucket 1)
        let result = seg.insert(0xF0, 0, 100, 1000);
        assert!(result.is_ok());
        let (bucket_idx, _) = result.unwrap();
        assert_eq!(bucket_idx, 1, "should be in neighbor bucket");

        // Fill rest of bucket 1 (11 more slots)
        for i in 0..(SLOTS_PER_BUCKET - 1) {
            let _ = seg.insert((0xF1 + i) as u8, 0, 101 + i as u32, 0);
        }

        // Next should go to stash (bucket 64+)
        let result = seg.insert(0xE0, 0, 200, 2000);
        assert!(result.is_ok());
        let (bucket_idx, _) = result.unwrap();
        assert!(
            bucket_idx >= REGULAR_BUCKETS,
            "should be in stash, got {}",
            bucket_idx
        );
    }

    #[test]
    fn test_stash_overflow() {
        let mut seg: Segment<u32, u32> = Segment::new(1);

        // With 2-bucket insert probing, fill home + neighbor = 24 slots
        for i in 0..24 {
            let result = seg.insert(i as u8, 0, i, i * 10);
            assert!(result.is_ok(), "Failed to insert {} (pre-stash)", i);
        }

        // Next insert should go to stash
        let result = seg.insert(0xFF, 0, 999, 9990);
        assert!(result.is_ok());
        let (bucket_idx, _) = result.unwrap();
        assert!(
            bucket_idx >= REGULAR_BUCKETS,
            "Should be in stash, got bucket {}",
            bucket_idx
        );
    }

    #[test]
    fn test_iter() {
        let mut seg: Segment<u32, u32> = Segment::new(1);

        let _ = seg.insert(0xAA, 0, 1, 10);
        let _ = seg.insert(0xBB, 100, 2, 20);
        let _ = seg.insert(0xCC, 200, 3, 30);

        let entries: Vec<_> = seg.iter().map(|(_, _, k, v)| (*k, *v)).collect();
        assert_eq!(entries.len(), 3);
        assert!(entries.contains(&(1, 10)));
        assert!(entries.contains(&(2, 20)));
        assert!(entries.contains(&(3, 30)));
    }

    #[test]
    fn test_clear() {
        let mut seg: Segment<u64, u64> = Segment::new(1);

        let _ = seg.insert(0xAA, 0, 1, 10);
        let _ = seg.insert(0xBB, 1, 2, 20);
        assert_eq!(seg.count(), 2);

        seg.clear();
        assert!(seg.is_empty());
        assert_eq!(seg.get(0xAA, 0, &1), None);
    }

    #[test]
    fn test_split() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut seg: Segment<u64, u64> = Segment::new(1);

        // Insert entries with known hash patterns
        for i in 0u64..20 {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let hash = hasher.finish();
            let fp = (hash & 0xFF) as u8;
            let _ = seg.insert(fp, hash, i, i * 100);
        }

        let initial_count = seg.count();
        assert_eq!(initial_count, 20);

        // Split the segment
        let sibling = seg.split(|key| {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            let hash = hasher.finish();
            let fp = (hash & 0xFF) as u8;
            (hash, fp)
        });

        // Both segments should have depth 2
        assert_eq!(seg.local_depth(), 2);
        assert_eq!(sibling.local_depth(), 2);

        // Total count should be preserved
        assert_eq!(seg.count() + sibling.count(), initial_count);
    }

    #[test]
    fn test_load_factor() {
        let mut seg: Segment<u32, u32> = Segment::new(1);

        assert_eq!(seg.load_factor(), 0.0);

        // Insert some entries
        for i in 0..100 {
            let _ = seg.insert(i as u8, i as u64, i, i * 10);
        }

        let expected = 100.0 / SEGMENT_CAPACITY as f64;
        assert!((seg.load_factor() - expected).abs() < 0.001);
    }

    #[test]
    fn test_split_rebuilds_stash_hints() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut seg: Segment<u64, u64> = Segment::new(1);

        let compute_hash = |key: &u64| -> (u64, u8) {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            let hash = hasher.finish();
            (hash, (hash & 0xFF) as u8)
        };

        let mut keys_in_stash = Vec::new();
        let mut inserted = 0;

        // Find keys that hash to bucket 0 and have split bit = 0
        for candidate in 0u64..10000 {
            let (hash, fp) = compute_hash(&candidate);
            let home = Segment::<u64, u64>::home_bucket(hash);
            let split_bit = (hash >> 62) & 1;

            if home == 0 && split_bit == 0 {
                let result = seg.insert(fp, hash, candidate, candidate * 10);
                if result.is_err() {
                    break;
                }
                let (bucket_idx, _) = result.unwrap();
                if bucket_idx >= REGULAR_BUCKETS {
                    keys_in_stash.push((candidate, hash, fp));
                }
                inserted += 1;
                if inserted >= 30 && !keys_in_stash.is_empty() {
                    break;
                }
            }
        }

        // Verify we have entries in stash
        assert!(
            !keys_in_stash.is_empty(),
            "Need stash entries for this test"
        );

        // Verify stash hints exist
        let has_hints_before =
            seg.bucket(0).has_stash_entries() || seg.bucket(1).has_stash_entries();
        assert!(has_hints_before, "Should have stash hints before split");

        // Split
        let _sibling = seg.split(|key| compute_hash(key));

        // Verify stash entries are still findable (hints were rebuilt)
        for (key, hash, fp) in &keys_in_stash {
            let found = seg.get(*fp, *hash, key);
            assert!(
                found.is_some(),
                "Lost stash entry {} after split (hints not rebuilt?)",
                key
            );
        }
    }
}
