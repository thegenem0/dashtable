//! Entry API for `DashTable`

use std::hash::{BuildHasher, Hash};

use crate::table::DashTable;

/// A view into a single entry in a `DashTable`, which may either be vacant or occupied
pub enum Entry<'a, K, V, S> {
    Occupied(OccupiedEntry<'a, K, V, S>),
    Vacant(VacantEntry<'a, K, V, S>),
}

/// A view into an occupied entry in a `DashTable`
pub struct OccupiedEntry<'a, K, V, S> {
    table: &'a mut DashTable<K, V, S>,
    key: K,
    hash: u64,
    fp: u8,
}

/// A view into a vacant entry in a `DashTable`
pub struct VacantEntry<'a, K, V, S> {
    table: &'a mut DashTable<K, V, S>,
    key: K,
    hash: u64,
    fp: u8,
}

impl<'a, K, V, S> Entry<'a, K, V, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher,
{
    /// Returns a reference to this entry's key
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => &entry.key,
            Entry::Vacant(entry) => &entry.key,
        }
    }

    /// Ensures a value is in the entry by inserting the default if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the
    /// default function if empty, and returns a mutable reference to
    /// the value in the entry
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the
    /// default function if empty
    /// The key is passed to the function
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the table
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(entry) = &mut self {
            f(entry.get_mut());
        }
        self
    }
}

impl<'a, K, V, S> Entry<'a, K, V, S>
where
    K: Eq + Hash + Clone,
    V: Default,
    S: BuildHasher,
{
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn or_default(self) -> &'a mut V {
        self.or_insert_with(V::default)
    }
}

impl<'a, K, V, S> OccupiedEntry<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub(crate) fn new(table: &'a mut DashTable<K, V, S>, key: K, hash: u64, fp: u8) -> Self {
        Self {
            table,
            key,
            hash,
            fp,
        }
    }

    /// Gets a reference to the key in the entry
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Gets a reference to the value in the entry
    pub fn get(&self) -> &V {
        self.table
            .directory
            .get(self.fp, self.hash, &self.key)
            .expect("OccupiedEntry: key not found")
    }

    /// Gets a mutable reference to the value in the entry
    pub fn get_mut(&mut self) -> &mut V {
        self.table
            .directory
            .get_mut(self.fp, self.hash, &self.key)
            .expect("OccupiedEntry: key not found")
    }

    /// Converts the entry into a mutable reference to its value
    pub fn into_mut(self) -> &'a mut V {
        self.table
            .directory
            .get_mut(self.fp, self.hash, &self.key)
            .expect("OccupiedEntry: key not found")
    }

    /// Sets the value of the entry, and returns the entry's old value
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }

    /// Takes the value out of the entry, and returns it
    pub fn remove(self) -> V {
        self.table
            .directory
            .remove(self.fp, self.hash, &self.key)
            .expect("OccupiedEntry: key not found")
    }
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher,
{
    pub(crate) fn new(table: &'a mut DashTable<K, V, S>, key: K, hash: u64, fp: u8) -> Self {
        Self {
            table,
            key,
            hash,
            fp,
        }
    }

    /// Gets a reference to the key that would be used when inserting
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Takes ownership of the key
    pub fn into_key(self) -> K {
        self.key
    }

    /// Sets the value of the entry, and returns a mutable reference to it
    pub fn insert(self, value: V) -> &'a mut V {
        let key_clone = self.key.clone();
        let hash = self.hash;
        let fp = self.fp;

        self.table.insert(self.key, value);

        self.table
            .directory
            .get_mut(fp, hash, &key_clone)
            .expect("VacantEntry::insert: failed to find inserted entry")
    }
}
