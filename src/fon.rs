#[cfg(test)]
pub mod tests;

use std::fmt::Debug;

use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
#[repr(transparent)]
pub struct Fon {
    id: u8,
}

pub const MAX_FON_ID: u8 = 127;

pub const MAX_FONS: usize = MAX_FON_ID as usize + 1;

pub const NO_FON: Fon = Fon { id: 0 };

pub const NO_FON_CHAR: char = '_';

#[derive(Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct FonSet {
    bits: u128,
}

impl From<u8> for Fon {
    fn from(id: u8) -> Self {
        debug_assert!(id <= MAX_FON_ID);
        Fon { id }
    }
}

impl From<usize> for Fon {
    fn from(i: usize) -> Self {
        debug_assert!(i < MAX_FONS);
        Fon { id: i as u8 }
    }
}

impl FonSet {
    pub const EMPTY: FonSet = FonSet { bits: 0 };

    pub fn new() -> FonSet {
        FonSet::EMPTY
    }

    pub fn contains(&self, fon: Fon) -> bool {
        self.bits & (1 << fon.id) != 0
    }

    pub fn len(&self) -> usize {
        self.bits.count_ones() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub fn is_real(&self) -> bool {
        !self.contains(NO_FON)
    }

    pub fn iter(&self) -> FonSetIter {
        FonSetIter {
            fonset: *self,
            index: 0,
        }
    }

    pub fn fons(&self, reg: &FonRegistry) -> Result<Vec<char>> {
        let mut result: Vec<char> = Vec::new();
        for i in self {
            result.push(reg.get_fon_char(i)?);
        }
        Ok(result)
    }

    pub fn seq_is_empty<S: AsRef<[FonSet]> + ?Sized>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.is_empty() || slice.contains(&FonSet::EMPTY)
    }

    pub fn seq_is_real<S: AsRef<[FonSet]> + ?Sized>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.is_empty() || slice.iter().all(|s| s.is_real())
    }

    pub fn seq_is_valid<S: AsRef<[FonSet]> + ?Sized>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.len() <= 1 || slice[1..slice.len() - 1].iter().all(|s| s.is_real())
    }

    pub fn first_fon(&self) -> Option<Fon> {
        self.iter().next()
    }

    pub fn next_fon_after(&self, mut fon: Fon) -> Option<Fon> {
        while fon.id < MAX_FON_ID {
            fon.id += 1;
            if self.contains(fon) {
                return Some(fon);
            }
        }
        None
    }

    pub fn seq_for_each_fon_seq<S, F>(seq: &S, mut action: F)
    where
        S: AsRef<[FonSet]> + ?Sized,
        F: FnMut(&[Fon]),
    {
        let slice = seq.as_ref();
        let mut buffer: Vec<Fon> = slice
            .iter()
            .map(|fonset| fonset.first_fon().unwrap())
            .collect();
        loop {
            action(&buffer);
            for i in (0..slice.len()).rev() {
                if let Some(fon) = slice[i].next_fon_after(buffer[i]) {
                    buffer[i] = fon;
                    break;
                } else if i == 0 {
                    return;
                } else {
                    buffer[i] = slice[i].first_fon().unwrap();
                }
            }
        }
    }

    pub fn seq_from_fonseq<S: IntoIterator<Item = Fon>>(seq: S) -> Vec<FonSet> {
        seq.into_iter().map(FonSet::from).collect()
    }

    pub fn seq_match_at_into<S: AsRef<[FonSet]>>(
        pattern: S,
        word: &[FonSet],
        word_idx: usize,
        result_buf: &mut Vec<FonSet>,
    ) -> bool {
        let pattern = pattern.as_ref();
        debug_assert!(word_idx + pattern.len() <= word.len());
        result_buf.clear();
        result_buf.extend_from_slice(word);
        for pattern_idx in 0..pattern.len() {
            result_buf[word_idx + pattern_idx] &= pattern[pattern_idx];
        }
        !FonSet::seq_is_empty(&result_buf[word_idx..word_idx + pattern.len()])
    }
}

impl From<Fon> for FonSet {
    fn from(fon: Fon) -> Self {
        FonSet { bits: 1 << fon.id }
    }
}

impl<const N: usize> From<[Fon; N]> for FonSet {
    fn from(fons: [Fon; N]) -> Self {
        let mut s = FonSet::new();
        for f in fons {
            s |= f;
        }
        s
    }
}

impl From<&[Fon]> for FonSet {
    fn from(fons: &[Fon]) -> Self {
        let mut s = FonSet::new();
        for f in fons {
            s |= *f;
        }
        s
    }
}

impl std::ops::BitOr for FonSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl std::ops::BitOr<Fon> for FonSet {
    type Output = Self;

    fn bitor(self, rhs: Fon) -> Self::Output {
        self | FonSet::from(rhs)
    }
}

impl std::ops::BitOrAssign for FonSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

impl std::ops::BitOrAssign<Fon> for FonSet {
    fn bitor_assign(&mut self, rhs: Fon) {
        *self |= FonSet::from(rhs)
    }
}

impl std::ops::BitAnd for FonSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits & rhs.bits,
        }
    }
}

impl std::ops::BitAndAssign for FonSet {
    fn bitand_assign(&mut self, rhs: Self) {
        self.bits &= rhs.bits;
    }
}

impl std::ops::Sub for FonSet {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits & !rhs.bits,
        }
    }
}

impl std::ops::Sub<Fon> for FonSet {
    type Output = Self;

    fn sub(self, rhs: Fon) -> Self::Output {
        self - FonSet::from(rhs)
    }
}

impl std::ops::SubAssign for FonSet {
    fn sub_assign(&mut self, rhs: Self) {
        self.bits &= !rhs.bits;
    }
}

impl std::ops::SubAssign<Fon> for FonSet {
    fn sub_assign(&mut self, rhs: Fon) {
        *self -= FonSet::from(rhs)
    }
}

impl Debug for FonSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter().map(|f| f.id)).finish()
    }
}

pub struct FonSetIter {
    fonset: FonSet,
    index: usize,
}

impl Iterator for FonSetIter {
    type Item = Fon;
    fn next(&mut self) -> Option<Fon> {
        if self.fonset.is_empty() {
            None
        } else {
            // skip all zeros, get nonzero index, then move one past
            let zeros = self.fonset.bits.trailing_zeros();
            self.index += zeros as usize;
            let item = Some(Fon {
                id: self.index as u8,
            });
            self.fonset.bits >>= zeros;
            self.fonset.bits >>= 1;
            self.index += 1;
            item
        }
    }
}

impl IntoIterator for FonSet {
    type Item = Fon;
    type IntoIter = FonSetIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for &FonSet {
    type Item = Fon;
    type IntoIter = FonSetIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Fon> for FonSet {
    fn from_iter<I: IntoIterator<Item = Fon>>(iter: I) -> Self {
        let mut s = FonSet::new();
        for i in iter {
            s |= i;
        }
        s
    }
}

impl<'a> FromIterator<&'a Fon> for FonSet {
    fn from_iter<I: IntoIterator<Item = &'a Fon>>(iter: I) -> Self {
        let mut s = FonSet::new();
        for &i in iter {
            s |= i;
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct FonRegistry {
    fons: Vec<char>,
}

impl FonRegistry {
    pub fn new() -> FonRegistry {
        FonRegistry {
            fons: vec![NO_FON_CHAR],
        }
    }

    pub fn add(&mut self, fon: char) -> Result<Fon> {
        match self.try_get_fon(fon) {
            Some(id) => Ok(id),
            None => {
                let id = self.fons.len();
                if id < MAX_FONS {
                    self.fons.push(fon);
                    Ok(Fon::from(id))
                } else {
                    Err(NoMoreFonIds)
                }
            }
        }
    }

    pub fn try_get_fon(&self, c: char) -> Option<Fon> {
        self.fons.iter().position(|&f| f == c).map(Fon::from)
    }

    pub fn get_fon(&self, c: char) -> Result<Fon> {
        self.try_get_fon(c).ok_or(NoSuchFon(c))
    }

    pub fn try_get_fon_char(&self, fon: Fon) -> Option<char> {
        self.fons.get(fon.id as usize).cloned()
    }

    pub fn get_fon_char(&self, fon: Fon) -> Result<char> {
        self.try_get_fon_char(fon).ok_or(NoSuchFonId(fon.id))
    }

    pub fn mark(&self) -> FonRegistryMark {
        FonRegistryMark(self.fons.len())
    }

    pub fn revert(&mut self, mark: FonRegistryMark) {
        self.fons.truncate(mark.0);
    }
}

impl Default for FonRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FonRegistryMark(usize);
