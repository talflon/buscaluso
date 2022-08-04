use thiserror::Error;

pub type Result<T> = std::result::Result<T, FonError>;

#[derive(Error, Clone, PartialEq, Eq, Debug)]
pub enum FonError {
    #[error("No such fon {0:?}")]
    NoSuchFon(char),
    #[error("Ran out of fon ids")]
    NoMoreFonIds,
    #[error("No such fon id {0}")]
    NoSuchFonId(FonId),
}

use FonError::*;

const MAX_FON_ID: FonId = 127;

type FonId = usize;

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
            let item = Some(self.index);
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
        self.fons.iter().position(|&f| f == fon)
    }

    pub fn get_id(&self, fon: char) -> Result<FonId> {
        self.try_get_id(fon).ok_or(NoSuchFon(fon))
    }

    pub fn try_get_fon(&self, id: FonId) -> Option<char> {
        self.fons.get(id).cloned()
    }

    pub fn get_fon(&self, id: FonId) -> Result<char> {
        self.try_get_fon(id).ok_or(NoSuchFonId(id))
    }
}

#[test]
fn test_fon_registry_empty() {
    let reg = FonRegistry::new();
    for c in ['q', 'é', '\0'] {
        assert_eq!(reg.get_id(c), Err(NoSuchFon(c)));
    }
    for i in [0, 1, 2, 35, MAX_FON_ID] {
        assert_eq!(reg.get_fon(i), Err(NoSuchFonId(i)));
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
    for c in (1..1000000u32)
        .flat_map(char::from_u32)
        .take((MAX_FON_ID as usize) + 1)
    {
        reg.add(c)?;
    }
    assert_eq!(reg.add('\0'), Err(NoMoreFonIds));
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
    let s: FonSet = [100, 99].into_iter().collect();
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
    let s1: FonSet = [0, 55, 3, 11, 8].into_iter().collect();
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
