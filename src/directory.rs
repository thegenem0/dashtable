use std::borrow::Borrow;

use crate::segment::{Segment, REGULAR_BUCKETS};

#[allow(dead_code)]
pub struct Directory<K, V> {
    /// Segment storage (owns segment)
    segments: Vec<Segment<K, V>>,

    /// Directory mapping hash prefixes to segment indices
    directory: Vec<usize>,

    /// Global depth (directory.len() == 2^global_depth)
    global_depth: u8,

    /// Total entries across all segments
    len: usize,
}

impl<K, V> Directory<K, V> {
    /// Create a new directory
    pub fn new() -> Self {
        Self::with_depth(0)
    }

    /// Create a new directory with given depth
    pub fn with_depth(initial_depth: u8) -> Self {
        let dir_size = 1usize << initial_depth;
        let mut segments = Vec::with_capacity(dir_size);
        let mut directory = Vec::with_capacity(dir_size);

        // init segments, one per directory
        for i in 0..dir_size {
            segments.push(Segment::new(initial_depth));
            directory.push(i);
        }

        Self {
            segments,
            directory,
            global_depth: initial_depth,
            len: 0,
        }
    }

    /// Number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if directory is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of unique segments
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Global depth
    #[inline]
    pub fn global_depth(&self) -> u8 {
        self.global_depth
    }

    /// Compute directory index from hash (uses HIGH bits)
    #[inline]
    fn dir_index(&self, hash: u64) -> usize {
        if self.global_depth == 0 {
            0
        } else {
            (hash >> (64 - self.global_depth)) as usize
        }
    }

    /// Get segment index for a given hash
    #[inline]
    fn segment_index(&self, hash: u64) -> usize {
        self.directory[self.dir_index(hash)]
    }

    /// Get segment for a given hash
    #[inline]
    fn get_segment(&self, hash: u64) -> &Segment<K, V> {
        let idx = self.segment_index(hash);
        &self.segments[idx]
    }

    /// Get mutable segment for a given hash
    #[inline]
    fn get_segment_mut(&mut self, hash: u64) -> &mut Segment<K, V> {
        let idx = self.segment_index(hash);
        &mut self.segments[idx]
    }

    /// Double the directory size
    fn grow_directory(&mut self) {
        // Each entry is duplicated in place: [A, B] -> [A, A, B, B]
        // This maintains the invariant that consecutive entries with
        // the same high-order bits share a segment
        let mut new_dir = Vec::with_capacity(self.directory.len() * 2);

        for &segment_idx in &self.directory {
            new_dir.push(segment_idx);
            new_dir.push(segment_idx);
        }

        self.directory = new_dir;
        self.global_depth += 1;
    }

    /// Find the first directory index pointing to a given segment
    fn first_dir_index_for_segment(&self, segment_idx: usize) -> usize {
        self.directory
            .iter()
            .position(|&idx| idx == segment_idx)
            .unwrap()
    }
}

impl<K, V> Directory<K, V>
where
    K: Eq,
{
    /// Get value by fingerprint, hash and key
    pub fn get<Q>(&self, fp: u8, hash: u64, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.get_segment(hash).get(fp, hash, key)
    }

    /// Get mutable value
    pub fn get_mut<Q>(&mut self, fp: u8, hash: u64, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.get_segment_mut(hash).get_mut(fp, hash, key)
    }

    /// Check if key exists
    pub fn contains_key<Q>(&self, fp: u8, hash: u64, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.get_segment(hash).contains(fp, hash, key)
    }

    /// Remove entry
    pub fn remove<Q>(&mut self, fp: u8, hash: u64, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let segment = self.get_segment_mut(hash);
        if let Some((_, value)) = segment.remove(fp, hash, key) {
            self.len -= 1;
            Some(value)
        } else {
            None
        }
    }

    /// Iterate over all key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.segments
            .iter()
            .flat_map(|seg| seg.iter().map(|(_, _, k, v)| (k, v)))
    }

    /// Iterate over all key-value pairs with mutable values
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.segments.iter_mut().flat_map(|seg| seg.iter_mut())
    }

    /// Iterate over all keys
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over all values
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over all values mutably
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }
}

impl<K, V> Directory<K, V>
where
    K: Eq + std::hash::Hash,
{
    /// Split segment and update directory
    pub fn split_segment<F>(&mut self, hash: u64, hash_fn: F)
    where
        F: FnMut(&K) -> (u64, u8),
    {
        let segment_idx = self.segment_index(hash);

        let needs_grow = self.segments[segment_idx].local_depth() == self.global_depth;
        if needs_grow {
            self.grow_directory();
        }

        // split segment
        let sibling = self.segments[segment_idx].split(hash_fn);
        let new_local_depth = sibling.local_depth(); // now old_local_depth + 1

        // add sibling to storage
        let sibling_idx = self.segments.len();
        self.segments.push(sibling);

        // update dir entries
        // half of entries pointing to segment_idx should now point to sibling_idx
        // entries to update are where distinguishing bit is 1
        let stride = 1usize << (self.global_depth - new_local_depth);
        let group_size = stride * 2; // num entries aliased

        // find firs dir entry for this segment
        let first_dir = self.first_dir_index_for_segment(segment_idx);

        // update every other group of `stride` entries, starting from `first_dir` + stride
        let mut i = first_dir + stride;
        while i < self.directory.len() {
            for j in 0..stride {
                if i + j < self.directory.len() && self.directory[i + j] == segment_idx {
                    self.directory[i + j] = sibling_idx;
                }
            }
            i += group_size;
        }
    }

    /// Insert key-value pair
    pub fn insert<F>(&mut self, fp: u8, hash: u64, key: K, value: V, mut hash_fn: F) -> Option<V>
    where
        F: FnMut(&K) -> (u64, u8) + Clone,
    {
        // check existing key
        {
            let segment_idx = self.segment_index(hash);
            if let Some((bucket_idx, slot_idx)) = self.segments[segment_idx].find(fp, hash, &key) {
                let bucket = if bucket_idx < REGULAR_BUCKETS {
                    self.segments[segment_idx].bucket_mut(bucket_idx)
                } else {
                    self.segments[segment_idx].stash_bucket_mut(bucket_idx - REGULAR_BUCKETS)
                };

                let old_value = std::mem::replace(unsafe { bucket.get_value_mut(slot_idx) }, value);

                return Some(old_value);
            }
        }

        // insert new entry and split if necessary
        let mut key = key;
        let mut value = value;

        loop {
            let segment_idx = self.segment_index(hash);
            match self.segments[segment_idx].insert(fp, hash, key, value) {
                Ok(_) => {
                    self.len += 1;
                    return None;
                }
                Err((k, v)) => {
                    key = k;
                    value = v;
                    self.split_segment(hash, &mut hash_fn);
                }
            }
        }
    }
}

impl<K, V> Default for Directory<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn compute_hash<K: Hash>(key: &K) -> (u64, u8) {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        (hash, (hash & 0xFF) as u8)
    }

    #[test]
    fn test_new_directory() {
        let dir: Directory<u64, u64> = Directory::new();
        assert!(dir.is_empty());
        assert_eq!(dir.segment_count(), 1);
        assert_eq!(dir.global_depth(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut dir: Directory<u64, u64> = Directory::new();

        let (hash, fp) = compute_hash(&42u64);
        let result = dir.insert(fp, hash, 42, 100, compute_hash);
        assert!(result.is_none()); // New insert
        assert_eq!(dir.len(), 1);

        let value = dir.get(fp, hash, &42);
        assert_eq!(value, Some(&100));
    }

    #[test]
    fn test_insert_update() {
        let mut dir: Directory<u64, u64> = Directory::new();

        let (hash, fp) = compute_hash(&42u64);
        dir.insert(fp, hash, 42, 100, compute_hash);
        let old = dir.insert(fp, hash, 42, 200, compute_hash);

        assert_eq!(old, Some(100));
        assert_eq!(dir.len(), 1);
        assert_eq!(dir.get(fp, hash, &42), Some(&200));
    }

    #[test]
    fn test_remove() {
        let mut dir: Directory<u64, u64> = Directory::new();

        let (hash, fp) = compute_hash(&42u64);
        dir.insert(fp, hash, 42, 100, compute_hash);

        let removed = dir.remove(fp, hash, &42);
        assert_eq!(removed, Some(100));
        assert!(dir.is_empty());
        assert_eq!(dir.get(fp, hash, &42), None);
    }

    #[test]
    fn test_split_on_insert() {
        let mut dir: Directory<u64, u64> = Directory::new();

        // Insert enough entries to trigger splits
        for i in 0u64..1000 {
            let (hash, fp) = compute_hash(&i);
            dir.insert(fp, hash, i, i * 10, compute_hash);
        }

        assert_eq!(dir.len(), 1000);
        assert!(
            dir.segment_count() > 1,
            "Should have split into multiple segments"
        );
        assert!(dir.global_depth() > 0, "Global depth should have increased");

        // Verify all entries are retrievable
        for i in 0u64..1000 {
            let (hash, fp) = compute_hash(&i);
            let value = dir.get(fp, hash, &i);
            assert_eq!(value, Some(&(i * 10)), "Missing entry {}", i);
        }
    }

    #[test]
    fn test_directory_growth() {
        let mut dir: Directory<u64, u64> = Directory::with_depth(1);

        assert_eq!(dir.global_depth(), 1);
        assert_eq!(dir.segment_count(), 2);

        // Fill segments to force growth
        for i in 0u64..2000 {
            let (hash, fp) = compute_hash(&i);
            dir.insert(fp, hash, i, i, compute_hash);
        }

        assert!(dir.global_depth() > 1);
    }

    #[test]
    fn test_borrow_lookup() {
        let mut dir: Directory<String, u64> = Directory::new();

        let key = "hello".to_string();
        let (hash, fp) = compute_hash(&key);
        dir.insert(fp, hash, key, 42, compute_hash);

        // Look up with &str instead of &String
        let value = dir.get(fp, hash, "hello");
        assert_eq!(value, Some(&42));

        // contains_key with &str
        assert!(dir.contains_key(fp, hash, "hello"));
        assert!(!dir.contains_key(fp, hash, "world"));

        // remove with &str
        let removed = dir.remove(fp, hash, "hello");
        assert_eq!(removed, Some(42));
    }
}
