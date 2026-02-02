//! Iterators for `DashTable`

use std::hash::{BuildHasher, Hash};

use crate::table::DashTable;

/// An iterator over the entries of a `DashTable`
pub struct Iter<'a, K, V> {
    inner: Box<dyn Iterator<Item = (&'a K, &'a V)> + 'a>,
}

impl<'a, K, V> Iter<'a, K, V> {
    pub(crate) fn new<I>(iter: I) -> Self
    where
        I: Iterator<Item = (&'a K, &'a V)> + 'a,
    {
        Self {
            inner: Box::new(iter),
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// A mutable iterator over the entries of a `DashTable`
pub struct IterMut<'a, K, V> {
    inner: Box<dyn Iterator<Item = (&'a K, &'a mut V)> + 'a>,
}

impl<'a, K, V> IterMut<'a, K, V> {
    pub(crate) fn new<I>(iter: I) -> Self
    where
        I: Iterator<Item = (&'a K, &'a mut V)> + 'a,
    {
        Self {
            inner: Box::new(iter),
        }
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// An iterator over the keys of a `DashTable`
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Keys<'a, K, V> {
    pub(crate) fn new(iter: Iter<'a, K, V>) -> Self {
        Self { inner: iter }
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

/// An iterator over the values of a `DashTable`
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Values<'a, K, V> {
    pub(crate) fn new(iter: Iter<'a, K, V>) -> Self {
        Self { inner: iter }
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

/// A mutable iterator over the values of a `DashTable`
pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}

impl<'a, K, V> ValuesMut<'a, K, V> {
    pub(crate) fn new(iter: IterMut<'a, K, V>) -> Self {
        Self { inner: iter }
    }
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

impl<'a, K, V, S> IntoIterator for &'a DashTable<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut DashTable<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
