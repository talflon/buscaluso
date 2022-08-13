use std::collections::BTreeMap;
use std::io;
use std::io::BufRead;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, FonError>;

#[derive(Error, Debug)]
pub enum FonError {
    #[error("No such fon {0:?}")]
    NoSuchFon(char),
    #[error("Ran out of fon ids")]
    NoMoreFonIds,
    #[error("No such fon id {0}")]
    NoSuchFonId(FonId),
    #[error("IO error {source:?}")]
    Io {
        #[from]
        source: io::Error,
    },
}

use FonError::*;

const MAX_FON_ID: FonId = 127;

type FonId = u8;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct FonSet {
    bits: u128,
}

impl FonSet {
    pub const NULL: FonSet = FonSet { bits: 0 };

    pub fn contains(&self, id: FonId) -> bool {
        self.bits & (1 << id) != 0
    }

    pub fn len(&self) -> usize {
        self.bits.count_ones() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub fn iter(&self) -> FonSetIter {
        FonSetIter {
            fonset: self.clone(),
            index: 0,
        }
    }

    pub fn fons(&self, reg: &FonRegistry) -> Result<Vec<char>> {
        let mut result: Vec<char> = Vec::new();
        for i in self.iter() {
            result.push(reg.get_fon(i)?);
        }
        Ok(result)
    }
}

impl From<FonId> for FonSet {
    fn from(id: FonId) -> Self {
        FonSet { bits: 1 << id }
    }
}

impl<const N: usize> From<[FonId; N]> for FonSet {
    fn from(fons: [FonId; N]) -> Self {
        let mut s = FonSet::NULL;
        for f in fons {
            s |= f;
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

impl std::ops::BitOr<FonId> for FonSet {
    type Output = Self;

    fn bitor(self, rhs: FonId) -> Self::Output {
        self | FonSet::from(rhs)
    }
}

impl std::ops::BitOrAssign for FonSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

impl std::ops::BitOrAssign<FonId> for FonSet {
    fn bitor_assign(&mut self, rhs: FonId) {
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

impl std::ops::Sub<FonId> for FonSet {
    type Output = Self;

    fn sub(self, rhs: FonId) -> Self::Output {
        self - FonSet::from(rhs)
    }
}

impl std::ops::SubAssign for FonSet {
    fn sub_assign(&mut self, rhs: Self) {
        self.bits &= !rhs.bits;
    }
}

impl std::ops::SubAssign<FonId> for FonSet {
    fn sub_assign(&mut self, rhs: FonId) {
        *self -= FonSet::from(rhs)
    }
}

pub struct FonSetIter {
    fonset: FonSet,
    index: usize,
}

impl Iterator for FonSetIter {
    type Item = FonId;
    fn next(&mut self) -> Option<FonId> {
        if self.fonset.is_empty() {
            None
        } else {
            // skip all zeros, get nonzero index, then move one past
            let zeros = self.fonset.bits.trailing_zeros();
            self.index += zeros as usize;
            let item = Some(self.index as FonId);
            self.fonset.bits >>= zeros;
            self.fonset.bits >>= 1;
            self.index += 1;
            item
        }
    }
}

impl IntoIterator for FonSet {
    type Item = FonId;
    type IntoIter = FonSetIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<FonId> for FonSet {
    fn from_iter<I: IntoIterator<Item = FonId>>(iter: I) -> Self {
        let mut s = FonSet::NULL;
        for i in iter {
            s |= i;
        }
        s
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FonSetSeq(Vec<FonSet>);

impl FonSetSeq {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty() || self.iter().any(|s| s.is_empty())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = FonSet> + '_ {
        self.0.iter().copied()
    }

    pub fn match_at(&self, seq: &FonSetSeq, index: usize) -> Option<FonSetSeq> {
        if index + self.len() > seq.len() {
            None
        } else {
            let mut result = seq.clone();
            for (i, s) in self.iter().enumerate() {
                result.0[index + i] &= s;
            }
            if result.is_empty() {
                None
            } else {
                Some(result)
            }
        }
    }
}

impl From<Vec<FonSet>> for FonSetSeq {
    fn from(v: Vec<FonSet>) -> FonSetSeq {
        FonSetSeq(v)
    }
}

impl<const N: usize> From<[FonSet; N]> for FonSetSeq {
    fn from(a: [FonSet; N]) -> FonSetSeq {
        FonSetSeq(Vec::from(a))
    }
}

#[derive(Clone, Debug)]
pub struct FonRegistry {
    fons: Vec<char>,
}

impl FonRegistry {
    pub fn new() -> FonRegistry {
        FonRegistry { fons: Vec::new() }
    }

    pub fn add(&mut self, fon: char) -> Result<FonId> {
        match self.try_get_id(fon) {
            Some(id) => Ok(id),
            None => {
                let id = self.fons.len() as FonId;
                if id <= MAX_FON_ID {
                    self.fons.push(fon);
                    Ok(id)
                } else {
                    Err(NoMoreFonIds)
                }
            }
        }
    }

    pub fn try_get_id(&self, fon: char) -> Option<FonId> {
        self.fons.iter().position(|&f| f == fon).map(|i| i as FonId)
    }

    pub fn get_id(&self, fon: char) -> Result<FonId> {
        self.try_get_id(fon).ok_or(NoSuchFon(fon))
    }

    pub fn try_get_fon(&self, id: FonId) -> Option<char> {
        self.fons.get(id as usize).cloned()
    }

    pub fn get_fon(&self, id: FonId) -> Result<char> {
        self.try_get_fon(id).ok_or(NoSuchFonId(id))
    }
}

trait Normalizer {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<FonId>) -> Result<()>;

    fn normalize(&self, word: &str) -> Result<Vec<FonId>> {
        let mut normalized = Vec::new();
        self.normalize_into(word, &mut normalized)?;
        Ok(normalized)
    }
}

impl Normalizer for FonRegistry {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<FonId>) -> Result<()> {
        for c in word.chars() {
            normalized.push(self.get_id(c)?);
        }
        Ok(())
    }
}

type Cost = i32;

pub struct BuscaCfg {
    fon_registry: FonRegistry,
    dictionary: BTreeMap<Box<[FonId]>, Vec<Box<str>>>,
}

impl BuscaCfg {
    pub fn new() -> BuscaCfg {
        BuscaCfg {
            fon_registry: FonRegistry::new(),
            dictionary: BTreeMap::new(),
        }
    }

    pub fn load_rules<R: BufRead>(&mut self, input: R) -> Result<()> {
        for line in input.lines() {
            let _s = line?;
        }
        Ok(())
    }

    pub fn add_to_dictionary(&mut self, word: &str, normalized: &[FonId]) -> Result<()> {
        match self.dictionary.get_mut(normalized) {
            Some(words) => words.push(word.into()),
            None => {
                self.dictionary.insert(normalized.into(), vec![word.into()]);
            }
        }
        Ok(())
    }

    pub fn load_dictionary<R: BufRead>(&mut self, mut input: R) -> Result<()> {
        let mut line = String::new();
        let mut normalized = Vec::new();
        while input.read_line(&mut line)? > 0 {
            let word = line.trim();
            self.normalize_into(word, &mut normalized)?;
            self.add_to_dictionary(word, &normalized)?;
            line.clear();
            normalized.clear();
        }
        Ok(())
    }

    pub fn words_iter(&self, fonseq: &[FonId]) -> impl Iterator<Item = &str> {
        let bs: &[Box<str>] = self.dictionary.get(fonseq).map_or(&[], Vec::as_slice);
        bs.iter().map(|s| &**s)
    }

    pub fn search(&self, word: &str) -> impl Iterator<Item = Result<(&str, Cost)>> {
        std::iter::empty()
    }
}

impl Normalizer for BuscaCfg {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<FonId>) -> Result<()> {
        self.fon_registry.normalize_into(word, normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_fon_registry_empty() {
        let reg = FonRegistry::new();
        for c in ['q', 'é', '\0'] {
            assert!(matches!(reg.get_id(c), Err(NoSuchFon(c_)) if c_ == c));
        }
        for i in [0, 1, 2, 35, MAX_FON_ID] {
            assert!(matches!(reg.get_fon(i), Err(NoSuchFonId(i_)) if i_ == i));
        }
    }

    #[test]
    fn test_fon_registry_add() -> Result<()> {
        let mut reg = FonRegistry::new();
        for c in ['Z', 'Ç'] {
            reg.add(c)?;
            assert_eq!(reg.get_fon(reg.get_id(c)?)?, c);
        }
        Ok(())
    }

    #[test]
    fn test_fon_registry_add_same() -> Result<()> {
        let mut reg = FonRegistry::new();
        let c = '$';
        let id = reg.add(c)?;
        for _ in 0..10000 {
            assert_eq!(reg.add(c)?, id);
        }
        Ok(())
    }

    #[test]
    fn test_too_many_registries() -> Result<()> {
        let mut reg = FonRegistry::new();
        for c in (1..)
            .flat_map(char::from_u32)
            .take((MAX_FON_ID as usize) + 1)
        {
            reg.add(c)?;
        }
        assert!(matches!(reg.add('\0'), Err(NoMoreFonIds)));
        Ok(())
    }

    #[test]
    fn test_fonset_null() {
        let s = FonSet::NULL;
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        for i in 0..=MAX_FON_ID {
            assert!(!s.contains(i))
        }
    }

    #[test]
    fn test_fonset_default() {
        let s: FonSet = Default::default();
        assert_eq!(s, FonSet::NULL);
    }

    #[test]
    fn test_fonset_from_one_contains_it() {
        for i in 0..=MAX_FON_ID {
            assert!(FonSet::from(i).contains(i));
        }
    }

    #[test]
    fn test_fonset_from_one_only_contains_it() {
        for i in 0..=MAX_FON_ID {
            let s = FonSet::from(i);
            assert_eq!(s.len(), 1);
            assert!(!s.is_empty());
            for j in 0..=MAX_FON_ID {
                if i != j {
                    assert!(!s.contains(j));
                }
            }
        }
    }

    #[test]
    fn test_fonset_add_with_oreq() {
        let mut s = FonSet::NULL;
        s |= 3;
        s |= 17;
        s |= 1;
        s |= 0;
        s |= 4;
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_fonset_subtract() {
        let s = FonSet::from([100, 99]);
        assert_eq!(s - 99, FonSet::from(100));
    }

    #[test]
    fn test_oreq_subeq() {
        let mut s = FonSet::NULL;
        s |= 77;
        s -= 77;
        assert!(!s.contains(77));
    }

    #[test]
    fn test_fonset_from_to_iter() {
        use std::collections::HashSet;
        let hs: HashSet<FonId> = [50, 7, 12, 4].into();
        let fs: FonSet = hs.iter().cloned().collect();
        let fshs: HashSet<FonId> = fs.iter().collect();
        assert_eq!(fshs, hs);
        assert!(fs.contains(12));
    }

    #[test]
    fn test_fonset_iter() {
        for i in 0..=MAX_FON_ID {
            let v: Vec<FonId> = FonSet::from(i).iter().collect();
            assert_eq!(v, vec![i]);
        }
    }

    #[test]
    fn test_fonset_for_loop() {
        let s1 = FonSet::from([0, 55, 3, 11, 8]);
        let mut s2: FonSet = FonSet::NULL;
        for i in s1 {
            s2 |= i;
        }
        assert_eq!(s2, s1);
    }

    #[test]
    fn test_fonset_fons() -> Result<()> {
        let mut reg = FonRegistry::new();
        let mut s = FonSet::NULL;
        let chars = vec!['$', 'q', 'A', 'ç'];
        for &c in chars.iter() {
            s |= reg.add(c)?;
        }
        assert_eq!(s.fons(&reg)?, chars);
        assert_eq!(FonSet::NULL.fons(&reg)?, vec![]);
        Ok(())
    }

    #[test]
    fn test_fonsetseq_empty() {
        assert!(FonSetSeq::from([]).is_empty());
        assert!(FonSetSeq::from([FonSet::NULL]).is_empty());
        assert!(FonSetSeq::from([FonSet::NULL, FonSet::NULL]).is_empty());
        assert!(FonSetSeq::from([FonSet::NULL, FonSet::from(3)]).is_empty());
        assert!(FonSetSeq::from([2.into(), FonSet::NULL]).is_empty());
        assert!(!FonSetSeq::from([2.into(), 3.into()]).is_empty());
    }

    #[test]
    fn test_fonsetseq_match_at_in_bounds() {
        let seq1 = FonSetSeq::from([[2, 4].into(), [1, 2, 3].into()]);
        let seq2 = FonSetSeq::from([[2, 5].into()]);
        assert_eq!(
            seq2.match_at(&seq1, 0),
            Some(FonSetSeq::from([2.into(), [1, 2, 3].into()]))
        );
        assert_eq!(
            seq2.match_at(&seq1, 1),
            Some(FonSetSeq::from([[2, 4].into(), 2.into()]))
        );
        assert_eq!(FonSetSeq::from([81.into()]).match_at(&seq1, 0), None);
        assert_eq!(FonSetSeq::from([81.into()]).match_at(&seq1, 1), None);
    }

    #[test]
    fn test_fonsetseq_match_at_out_of_bounds() {
        assert_eq!(FonSetSeq::from([]).match_at(&FonSetSeq::from([]), 1), None);
        assert_eq!(
            FonSetSeq::from([[32, 0].into()]).match_at(&FonSetSeq::from([[90, 0, 1].into()]), 1),
            None
        );
    }

    #[test]
    fn test_registry_normalize() -> Result<()> {
        let mut reg = FonRegistry::new();
        let s1 = "test";
        let s2 = "blah";
        for c in s1.chars().chain(s2.chars()) {
            reg.add(c)?;
        }
        let mut n1a = Vec::new();
        reg.normalize_into(s1, &mut n1a)?;
        assert!(!n1a.is_empty());
        let mut n1b = Vec::new();
        reg.normalize_into(&"a test!"[2..6], &mut n1b)?;
        assert_eq!(n1b, n1a);
        let mut n2 = Vec::new();
        reg.normalize_into(s2, &mut n2)?;
        assert_ne!(n2, n1a);
        Ok(())
    }

    #[test]
    fn test_registry_normalize_bad_chars() {
        let reg = FonRegistry::new();
        assert!(matches!(reg.normalize("hi"), Err(NoSuchFon(_))));
    }

    #[test]
    fn test_dictionary_one_key() -> Result<()> {
        let mut cfg = BuscaCfg::new();
        assert_eq!(Vec::from_iter(cfg.words_iter(b"one")), &[] as &[&str]);

        cfg.add_to_dictionary("first", b"one")?;
        assert_eq!(Vec::from_iter(cfg.words_iter(b"one")), &["first"]);

        cfg.add_to_dictionary("another", b"one")?;
        assert_eq!(
            BTreeSet::from_iter(cfg.words_iter(b"one")),
            BTreeSet::from(["another", "first"])
        );
        Ok(())
    }

    #[test]
    fn test_dictionary_two_keys() -> Result<()> {
        let mut cfg = BuscaCfg::new();
        cfg.add_to_dictionary("first", b"one")?;
        cfg.add_to_dictionary("another", b"one")?;
        assert_eq!(Vec::from_iter(cfg.words_iter(b"two")), &[] as &[&str]);

        cfg.add_to_dictionary("word", b"two")?;
        assert_eq!(Vec::from_iter(cfg.words_iter(b"two")), &["word"]);

        assert_eq!(
            BTreeSet::from_iter(cfg.words_iter(b"one")),
            BTreeSet::from(["another", "first"])
        );
        Ok(())
    }
}
