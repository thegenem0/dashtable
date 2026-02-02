use std::{
    borrow::Borrow,
    hash::{BuildHasher, Hash, RandomState},
};

use crate::{
    directory::Directory,
    entry::{Entry, OccupiedEntry, VacantEntry},
    iter::{Iter, IterMut, Keys, Values, ValuesMut},
};

/// A high-performance hash table using extendible hashing with segment-based growth
///
/// `DashTable` is designed for cache-friendly operations and efficient memory usage
pub struct DashTable<K, V, S = RandomState> {
    pub(crate) directory: Directory<K, V>,
    hash_builder: S,
}

impl<K, V> DashTable<K, V, RandomState> {
    /// Create an empty `DashTable`
    #[inline]
    pub fn new() -> Self {
        Self::with_hasher(RandomState::new())
    }
}

impl<K, V, S> DashTable<K, V, S> {
    /// Create an empty `DashTable` with provided hasher
    #[inline]
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            directory: Directory::new(),
            hash_builder,
        }
    }

    /// Returns the number of elements in the table
    #[inline]
    pub fn len(&self) -> usize {
        self.directory.len()
    }

    /// Returns `true` if the table contains no elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.directory.is_empty()
    }

    /// Returns a reference to the hasher
    #[inline]
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Returns the number of segments
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.directory.segment_count()
    }

    /// Returns the global depth of the directory
    #[inline]
    pub fn global_depth(&self) -> u8 {
        self.directory.global_depth()
    }
}

impl<K, V, S> DashTable<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Computes hash and fingerprint for a given key
    #[inline]
    fn hash_and_fingerprint<Q>(&self, key: &Q) -> (u64, u8)
    where
        Q: ?Sized + Hash,
    {
        let hash = self.hash_builder.hash_one(key);
        let fp = (hash & 0xFF) as u8;
        (hash, fp)
    }

    /// Returns a reference to the value associated with the given key
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (hash, fp) = self.hash_and_fingerprint(key);
        self.directory.get(fp, hash, key)
    }

    /// Returns a mutable reference to the value associated with the given key
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (hash, fp) = self.hash_and_fingerprint(key);
        self.directory.get_mut(fp, hash, key)
    }

    /// Inserts a key-value pair into the table
    ///
    /// If the table did not have this key present, `None` is returned.
    /// If the table did have this key present, the value is updated, and the old value is returned.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (hash, fp) = self.hash_and_fingerprint(&key);
        let hash_builder = &self.hash_builder;

        self.directory.insert(fp, hash, key, value, |k| {
            let h = hash_builder.hash_one(k);
            (h, (h & 0xFF) as u8)
        })
    }

    /// Removes a ke from the table, returning the value if the key was present
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (hash, fp) = self.hash_and_fingerprint(key);
        self.directory.remove(fp, hash, key)
    }

    /// Return `true` if the table contains a value for the given key
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (hash, fp) = self.hash_and_fingerprint(key);
        self.directory.get(fp, hash, key).is_some()
    }

    /// Clears the table, removing all key-value pairs
    #[inline]
    pub fn clear(&mut self) {
        self.directory = Directory::new();
    }

    /// Returns an iterator over all key-value pairs
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self.directory.iter())
    }

    /// Returns a mutable iterator over all key-value pairs
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut::new(self.directory.iter_mut())
    }

    /// Returns an iterator over all keys
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys::new(self.iter())
    }

    /// Returns an iterator over all values
    pub fn values(&self) -> Values<'_, K, V> {
        Values::new(self.iter())
    }

    /// Returns a mutable iterator over all values
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut::new(self.iter_mut())
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S>
    where
        K: Clone,
    {
        let (hash, fp) = self.hash_and_fingerprint(&key);

        if self.directory.contains_key(fp, hash, &key) {
            Entry::Occupied(OccupiedEntry::new(self, key, hash, fp))
        } else {
            Entry::Vacant(VacantEntry::new(self, key, hash, fp))
        }
    }
}

impl<K, V> Default for DashTable<K, V, RandomState> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let table: DashTable<u64, u64> = DashTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_get() {
        let mut table = DashTable::new();

        table.insert(1u64, 100u64);
        table.insert(2u64, 200u64);

        assert_eq!(table.get(&1), Some(&100));
        assert_eq!(table.get(&2), Some(&200));
        assert_eq!(table.get(&3), None);
    }

    #[test]
    fn test_get_mut() {
        let mut table = DashTable::new();

        table.insert(1u64, 100u64);

        if let Some(v) = table.get_mut(&1) {
            *v = 999;
        }

        assert_eq!(table.get(&1), Some(&999));
    }

    #[test]
    fn test_get_with_borrow() {
        let mut table = DashTable::new();

        table.insert("hello".to_string(), 42);

        // Look up with &str instead of &String
        assert_eq!(table.get("hello"), Some(&42));
        assert_eq!(table.get("world"), None);
    }

    #[test]
    fn test_insert_and_get() {
        let mut table = DashTable::new();

        assert!(table.insert(1u64, 100u64).is_none());
        assert_eq!(table.len(), 1);

        assert!(table.contains_key(&1));
        assert!(!table.contains_key(&2));
    }

    #[test]
    fn test_insert_update() {
        let mut table = DashTable::new();

        table.insert(1u64, 100u64);
        let old = table.insert(1u64, 200u64);

        assert_eq!(old, Some(100));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut table = DashTable::new();

        table.insert(1u64, 100u64);
        let removed = table.remove(&1);

        assert_eq!(removed, Some(100));
        assert!(table.is_empty());
        assert!(!table.contains_key(&1));
    }

    #[test]
    fn test_clear() {
        let mut table = DashTable::new();

        for i in 0u64..100 {
            table.insert(i, i * 10);
        }

        assert_eq!(table.len(), 100);
        table.clear();
        assert!(table.is_empty());
    }

    #[test]
    fn test_many_inserts() {
        let mut table = DashTable::new();

        for i in 0u64..10_000 {
            table.insert(i, i * 10);
        }

        assert_eq!(table.len(), 10_000);

        // Verify all entries
        for i in 0u64..10_000 {
            assert!(table.contains_key(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_string_keys() {
        let mut table = DashTable::new();

        table.insert("hello".to_string(), 1);
        table.insert("world".to_string(), 2);

        // both string and &str work
        assert!(table.contains_key("hello"));
        assert!(table.contains_key("world"));
        assert!(!table.contains_key(&"foo".to_string()));
    }

    #[test]
    fn test_custom_hasher() {
        use std::collections::hash_map::RandomState;

        let hasher = RandomState::new();
        let mut table: DashTable<u64, u64, _> = DashTable::with_hasher(hasher);

        table.insert(1, 100);
        assert!(table.contains_key(&1));
    }

    #[test]
    fn test_iter() {
        let mut table = DashTable::new();

        for i in 0u64..100 {
            table.insert(i, i * 10);
        }

        let mut count = 0;
        for (k, v) in table.iter() {
            assert_eq!(*v, *k * 10);
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn test_iter_mut() {
        let mut table = DashTable::new();

        for i in 0u64..100 {
            table.insert(i, i);
        }

        for (_, v) in table.iter_mut() {
            *v *= 2;
        }

        for i in 0u64..100 {
            assert_eq!(table.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_keys_values() {
        let mut table = DashTable::new();

        table.insert(1u64, 10u64);
        table.insert(2, 20);
        table.insert(3, 30);

        let keys: Vec<_> = table.keys().copied().collect();
        let values: Vec<_> = table.values().copied().collect();

        assert_eq!(keys.len(), 3);
        assert_eq!(values.len(), 3);

        // Check all keys and values are present (order may vary)
        assert!(keys.contains(&1) && keys.contains(&2) && keys.contains(&3));
        assert!(values.contains(&10) && values.contains(&20) && values.contains(&30));
    }

    #[test]
    fn test_for_loop() {
        let mut table = DashTable::new();

        table.insert("a".to_string(), 1);
        table.insert("b".to_string(), 2);

        let mut sum = 0;
        for (_, v) in &table {
            sum += v;
        }
        assert_eq!(sum, 3);
    }

    #[test]
    fn test_entry_or_insert() {
        let mut table = DashTable::new();

        table.entry(1u64).or_insert(10u64);
        assert_eq!(table.get(&1), Some(&10));

        table.entry(1u64).or_insert(20u64);
        assert_eq!(table.get(&1), Some(&10)); // Should not change
    }

    #[test]
    fn test_entry_or_insert_with() {
        let mut table = DashTable::new();

        table.entry(1u64).or_insert_with(|| 100u64);
        assert_eq!(table.get(&1), Some(&100));
    }

    #[test]
    fn test_entry_and_modify() {
        let mut table = DashTable::new();

        table.insert(1u64, 10u64);

        table.entry(1).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(table.get(&1), Some(&15));

        table.entry(2).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(table.get(&2), Some(&0));
    }

    #[test]
    fn test_entry_or_default() {
        let mut table: DashTable<u64, u64> = DashTable::new();

        *table.entry(1).or_default() += 1;
        *table.entry(1).or_default() += 1;

        assert_eq!(table.get(&1), Some(&2));
    }

    #[test]
    fn test_occupied_entry() {
        let mut table = DashTable::new();
        table.insert(1u64, 10u64);

        if let Entry::Occupied(mut entry) = table.entry(1) {
            assert_eq!(entry.key(), &1);
            assert_eq!(entry.get(), &10);

            *entry.get_mut() = 20;
            assert_eq!(entry.get(), &20);

            let old = entry.insert(30);
            assert_eq!(old, 20);
        } else {
            panic!("Expected occupied entry");
        }

        assert_eq!(table.get(&1), Some(&30));
    }

    #[test]
    fn test_vacant_entry() {
        let mut table: DashTable<u64, u64> = DashTable::new();

        if let Entry::Vacant(entry) = table.entry(1) {
            assert_eq!(entry.key(), &1);
            let value = entry.insert(100);
            *value += 1;
        } else {
            panic!("Expected vacant entry");
        }

        assert_eq!(table.get(&1), Some(&101));
    }

    #[test]
    fn test_entry_remove() {
        let mut table = DashTable::new();
        table.insert(1u64, 10u64);

        if let Entry::Occupied(entry) = table.entry(1) {
            let value = entry.remove();
            assert_eq!(value, 10);
        }

        assert!(table.get(&1).is_none());
    }
}
