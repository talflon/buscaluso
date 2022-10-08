// Copyright © 2022 Daniel Getz
// SPDX-License-Identifier: MIT

use super::*;
use crate::tests::*;

use std::cmp::min;
use std::collections::BTreeSet;

use quickcheck::{Arbitrary, QuickCheck, TestResult};
use quickcheck_macros::*;

impl Arbitrary for Fon {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        Fon {
            id: FonId::arbitrary(g) % MAX_FON_ID,
        }
    }
}

impl Arbitrary for FonSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        FonSet {
            bits: FonSetBitSet::arbitrary(g),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmallFonSet(FonSet);

impl Arbitrary for SmallFonSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        SmallFonSet(FonSet {
            bits: u16::arbitrary(g) as FonSetBitSet,
        })
    }
}

fn unwrap_small_seq(seq: impl AsRef<[SmallFonSet]>) -> Vec<FonSet> {
    seq.as_ref().iter().map(|sfs| sfs.0).collect()
}

#[derive(Debug, Clone)]
pub struct NonEmptyFonSetSeq(Vec<FonSet>);

impl Arbitrary for NonEmptyFonSetSeq {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        loop {
            let v = Vec::arbitrary(g);
            if !v.is_empty() && !v.contains(&FonSet::EMPTY) {
                return NonEmptyFonSetSeq(v);
            }
        }
    }
}

impl AsRef<[FonSet]> for NonEmptyFonSetSeq {
    fn as_ref(&self) -> &[FonSet] {
        self.0.as_ref()
    }
}

impl IntoIterator for NonEmptyFonSetSeq {
    type Item = FonSet;

    type IntoIter = <Vec<FonSet> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

pub trait TestingFonRegistry {
    fn set(&mut self, chars: &str) -> Result<FonSet>;
    fn seq(&mut self, chars: &str) -> Result<Vec<Fon>>;
    fn setseq(&mut self, sets: &[&str]) -> Result<Vec<FonSet>>;
}

impl TestingFonRegistry for FonRegistry {
    fn set(&mut self, chars: &str) -> Result<FonSet> {
        let mut s = FonSet::new();
        for c in chars.chars() {
            s |= self.add(c)?;
        }
        Ok(s)
    }

    fn seq(&mut self, chars: &str) -> Result<Vec<Fon>> {
        let mut s = Vec::new();
        for c in chars.chars() {
            s.push(self.add(c)?);
        }
        Ok(s)
    }

    fn setseq(&mut self, sets: &[&str]) -> Result<Vec<FonSet>> {
        let mut seq = Vec::new();
        for s in sets {
            seq.push(self.set(s)?);
        }
        Ok(seq)
    }
}

#[test]
fn test_fon_registry_empty() {
    let reg = FonRegistry::new();
    for c in ['q', 'é', '\0'] {
        assert!(matches!(reg.get_fon(c), Err(NoSuchFon(c_)) if c_ == c));
    }
    for i in [1, 2, 35, MAX_FON_ID] {
        assert!(matches!(reg.get_fon_char(Fon::from(i)), Err(NoSuchFonId(i_)) if i_ == i.into()));
    }
}

#[test]
fn test_fon_registry_has_no_fon() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(reg.try_get_fon(NO_FON_CHAR), Some(NO_FON));
    assert_eq!(reg.try_get_fon_char(NO_FON), Some(NO_FON_CHAR));
    reg.add('ȟ')?;
    reg.add('r')?;
    assert_eq!(reg.try_get_fon(NO_FON_CHAR), Some(NO_FON));
    assert_eq!(reg.try_get_fon_char(NO_FON), Some(NO_FON_CHAR));
    Ok(())
}

#[quickcheck]
fn test_fon_registry_add(chars: Vec<char>) -> Result<TestResult> {
    if chars.len() > MAX_FONS - 1 {
        return Ok(TestResult::discard());
    }
    let mut reg = FonRegistry::new();
    for c in chars {
        reg.add(c)?;
        assert_eq!(reg.get_fon_char(reg.get_fon(c)?)?, c);
    }
    Ok(TestResult::passed())
}

#[test]
fn test_fon_registry_add_same() -> Result<()> {
    let mut reg = FonRegistry::new();
    let c = '$';
    let fon = reg.add(c)?;
    for _ in 0..10000 {
        assert_eq!(reg.add(c)?, fon);
    }
    Ok(())
}

#[test]
fn test_too_many_registries() -> Result<()> {
    let mut reg = FonRegistry::new();
    for c in (1..)
        .flat_map(char::from_u32)
        .filter(|&c| c != NO_FON_CHAR)
        .take(MAX_FON_ID as usize)
    {
        reg.add(c)?;
    }
    assert!(matches!(reg.add('\0'), Err(NoMoreFonIds)));
    Ok(())
}

#[test]
fn test_fonset_empty() {
    let s = FonSet::EMPTY;
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
    for i in 0..=MAX_FON_ID {
        assert!(!s.contains(Fon::from(i)))
    }
}

#[test]
fn test_fonset_real() {
    let mut s = FonSet::new();
    assert!(s.is_real());
    s |= Fon::from(12u8);
    assert!(s.is_real());
    s |= NO_FON;
    assert!(!s.is_real());
    s -= Fon::from(12u8);
    assert!(!s.is_real());
    s |= Fon::from(3u8);
    s |= Fon::from(61u8);
    assert!(!s.is_real());
    s -= NO_FON;
    assert!(s.is_real());
}

#[test]
fn test_fonset_default() {
    let s: FonSet = Default::default();
    assert_eq!(s, FonSet::EMPTY);
}

#[test]
fn test_fonset_new() {
    let s: FonSet = FonSet::new();
    assert_eq!(s, FonSet::EMPTY);
}

#[test]
fn test_fonset_from_one_contains_it() {
    for i in 0..=MAX_FON_ID {
        assert!(FonSet::from(Fon::from(i)).contains(Fon::from(i)));
    }
}

#[test]
fn test_fonset_from_one_only_contains_it() {
    for i in 0..=MAX_FON_ID {
        let s = FonSet::from(Fon::from(i));
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
        for j in 0..=MAX_FON_ID {
            if i != j {
                assert!(!s.contains(Fon::from(j)));
            }
        }
    }
}

#[quickcheck]
fn test_fonset_or_fon(set: FonSet, fon: Fon) {
    assert!((set | fon).contains(fon));
    assert_eq!(set | fon, set | FonSet::from(fon));
}

#[quickcheck]
fn test_fonset_oreq_fon(set: FonSet, fon: Fon) {
    let mut set_copy = set;
    set_copy |= fon;
    assert_eq!(set_copy, set | fon);
}

#[quickcheck]
fn test_fonset_or(set1: FonSet, set2: FonSet) {
    let uni = set1 | set2;
    for id in 0..=MAX_FON_ID {
        let fon = Fon::from(id);
        assert_eq!(set1.contains(fon) || set2.contains(fon), uni.contains(fon));
    }
}

#[quickcheck]
fn test_fonset_oreq(set: FonSet, other: FonSet) {
    let mut set_copy = set;
    set_copy |= other;
    assert_eq!(set_copy, set | other);
}

#[quickcheck]
fn test_fonset_sub_fon(set: FonSet, fon: Fon) {
    assert!(!(set - fon).contains(fon));
    assert_eq!(set - fon, set - FonSet::from(fon));
}

#[quickcheck]
fn test_fonset_subeq_fon(set: FonSet, fon: Fon) {
    let mut set_copy = set;
    set_copy -= fon;
    assert_eq!(set_copy, set - fon);
}

#[quickcheck]
fn test_fonset_is_subset_of_self(set: FonSet) -> bool {
    set.is_subset_of(set)
}

#[quickcheck]
fn test_fonset_is_subset_of_same_as_contains_of_itered(set1: SmallFonSet, set2: SmallFonSet) {
    let set1 = set1.0;
    let set2 = set2.0;
    assert_eq!(
        set1.is_subset_of(set2),
        set1.into_iter().all(|fon| set2.contains(fon))
    );
}

#[quickcheck]
fn test_fonset_is_subset_of(mut set1: FonSet, set2: FonSet) -> TestResult {
    set1 |= set2;
    if set1 == set2 {
        return TestResult::discard();
    }
    assert!(set2.is_subset_of(set1));
    assert!(!set1.is_subset_of(set2));
    TestResult::passed()
}

#[quickcheck]
fn test_fonsetseq_is_subset_of_seq_same_length(
    seq1: Vec<SmallFonSet>,
    seq2: Vec<SmallFonSet>,
) -> TestResult {
    let len = min(seq1.len(), seq2.len());
    if len == 0 {
        return TestResult::discard();
    }
    let seq1 = unwrap_small_seq(&seq1[..len]);
    let seq2 = unwrap_small_seq(&seq2[..len]);
    if seq1 == seq2 {
        return TestResult::discard();
    }
    let unioned: Vec<FonSet> = seq1.iter().zip(seq2.iter()).map(|(&x, &y)| x | y).collect();
    assert!(seq1.is_subset_of_seq(&unioned));
    assert!(seq2.is_subset_of_seq(&unioned));
    assert!(unioned.is_subset_of_seq(&seq2) == seq1.is_subset_of_seq(&seq2));
    assert!(unioned.is_subset_of_seq(&seq1) == seq2.is_subset_of_seq(&seq1));
    TestResult::passed()
}

#[quickcheck]
fn test_fonsetseq_is_subset_of_seq_different_length(seq: Vec<SmallFonSet>) -> TestResult {
    if seq.is_empty() {
        return TestResult::discard();
    }
    let seq = unwrap_small_seq(seq);
    assert!(!seq.is_subset_of_seq(&seq[1..]));
    assert!(!seq.is_subset_of_seq(&seq[..seq.len() - 1]));
    TestResult::passed()
}

#[quickcheck]
fn test_fonsetseq_is_subset_of_seq_self(seq: Vec<FonSet>) -> bool {
    seq.is_subset_of_seq(&seq)
}

#[quickcheck]
fn test_fonset_to_from_iter(fs: FonSet) {
    use std::collections::HashSet;
    let hs: HashSet<Fon> = fs.iter().collect();
    assert_eq!(FonSet::from_iter(hs.iter()), fs);
}

#[quickcheck]
fn test_fonset_iter(fon_ids: BTreeSet<Fon>) {
    assert!(FonSet::from_iter(fon_ids.iter())
        .iter()
        .eq(fon_ids.iter().cloned()));
}

#[quickcheck]
fn test_fonset_for_loop(s1: FonSet) {
    let mut s2: FonSet = FonSet::new();
    for i in s1 {
        s2 |= i;
    }
    assert_eq!(s2, s1);
}

#[test]
fn test_fonset_fons() -> Result<()> {
    let mut reg = FonRegistry::new();
    let mut s = FonSet::new();
    let chars = vec!['$', 'q', 'A', 'ç'];
    for &c in chars.iter() {
        s |= reg.add(c)?;
    }
    assert_eq!(s.fons(&reg)?, chars);
    assert!(FonSet::EMPTY.fons(&reg)?.is_empty());
    Ok(())
}

#[quickcheck]
fn test_fonsetseq_empty_if_any_empty(setseq: WithInsIndex<Vec<FonSet>>) {
    let index = setseq.index;
    let mut setseq = setseq.item;
    setseq.insert(index, FonSet::EMPTY);
    assert!(setseq.is_empty_seq());
}

#[quickcheck]
fn test_fonsetseq_not_empty(setseq: NonEmptyFonSetSeq) -> bool {
    !setseq.is_empty_seq()
}

#[test]
fn test_fonsetseq_zero_len_empty() {
    assert!([].is_empty_seq());
}

#[test]
fn test_fonsetseq_real() {
    assert!([].is_real_seq());
    assert!([FonSet::EMPTY].is_real_seq());
    assert!(![FonSet::from(NO_FON)].is_real_seq());
    assert!(![
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::from(NO_FON),
        FonSet::from(Fon::from(50u8)),
    ]
    .is_real_seq());
    assert!([
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::EMPTY,
        FonSet::from(Fon::from(50u8)),
    ]
    .is_real_seq());
}

#[test]
fn test_fonsetseq_valid() {
    assert!([].is_valid_seq());
    assert!([FonSet::EMPTY].is_valid_seq());
    assert!([FonSet::from(NO_FON)].is_valid_seq());
    assert!(![
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::from(NO_FON),
        FonSet::from(Fon::from(50u8)),
    ]
    .is_valid_seq());
    assert!([
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::EMPTY,
        FonSet::from(Fon::from(50u8)),
    ]
    .is_valid_seq());
    assert!(&[
        FonSet::from(NO_FON),
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::from(Fon::from(50u8)),
        FonSet::from(NO_FON)
    ]
    .is_valid_seq());
}

#[test]
fn test_fonset_empty_first_fon() {
    assert_eq!(FonSet::EMPTY.first_fon(), None);
}

#[quickcheck]
fn test_fonset_len1_first_fon(fon: Fon) {
    assert_eq!(FonSet::from(fon).first_fon(), Some(fon));
}

#[quickcheck]
fn test_fonset_first_fon(set: FonSet) {
    assert_eq!(set.first_fon(), set.iter().next());
}

#[quickcheck]
fn test_fonset_next_fon_after(set: FonSet) -> TestResult {
    if set.is_empty() {
        return TestResult::discard();
    }
    let mut iter = set.iter();
    let mut last = iter.next().unwrap();
    for next in iter {
        if set.next_fon_after(last) != Some(next) {
            return TestResult::failed();
        }
        last = next;
    }
    TestResult::passed()
}

#[quickcheck]
fn test_fonset_next_fon_after_same_as_iter(fonset: FonSet) {
    let by_iter: Vec<Fon> = fonset.iter().collect();
    let mut by_next_fon_after = Vec::new();
    let mut current = fonset.first_fon();
    while let Some(id) = current {
        by_next_fon_after.push(id);
        current = fonset.next_fon_after(id);
    }
    assert_eq!(by_next_fon_after, by_iter);
}

#[test]
fn test_fonset_seq_for_each_fon_seq() -> Result<()> {
    let mut reg = FonRegistry::new();
    let mut seqs: Vec<Vec<Fon>> = Vec::new();
    reg.setseq(&["a"])?
        .for_each_fon_seq(&mut Vec::new(), |s| seqs.push(s.into()));
    assert_eq!(seqs, vec![reg.seq("a")?]);
    seqs.clear();
    reg.setseq(&["x", "abc", "bx"])?
        .for_each_fon_seq(&mut Vec::new(), |s| seqs.push(s.into()));
    let mut expected = vec![
        reg.seq("xab")?,
        reg.seq("xbb")?,
        reg.seq("xcb")?,
        reg.seq("xax")?,
        reg.seq("xbx")?,
        reg.seq("xcx")?,
    ];
    seqs.sort();
    expected.sort();
    assert_eq!(seqs, expected);
    Ok(())
}

#[test]
fn test_fonset_seq_for_each_fon_seq_len() {
    fn fonset_seq_for_each_fon_seq_len(seq: Vec<SmallFonSet>) -> TestResult {
        let seq = Vec::from_iter(seq.iter().map(|SmallFonSet(s)| *s));
        let expected_len: usize = seq.iter().map(FonSet::len).product();
        if seq.is_empty() || expected_len == 0 || expected_len > 10_000 {
            TestResult::discard()
        } else {
            let mut count: usize = 0;
            seq.for_each_fon_seq(&mut Vec::new(), |_| {
                count += 1;
            });
            TestResult::from_bool(count == expected_len)
        }
    }
    QuickCheck::new()
        .gen(quickcheck::Gen::new(8))
        .quickcheck(fonset_seq_for_each_fon_seq_len as fn(_) -> TestResult);
}

#[test]
fn test_fonset_seq_for_each_fon_seq_contains_samples() {
    fn fonset_seq_for_each_fon_seq_contains_samples(seq: Vec<SmallFonSet>) -> TestResult {
        let seq = Vec::from_iter(seq.iter().map(|SmallFonSet(s)| *s));
        let expected_len: usize = seq.iter().map(FonSet::len).product();
        if seq.is_empty() || expected_len == 0 || expected_len > 10_000 {
            TestResult::discard()
        } else {
            let mut looking_for = BTreeSet::new();
            looking_for.insert(Vec::from_iter(seq.iter().map(|s| s.first_fon().unwrap())));
            looking_for.insert(Vec::from_iter(
                seq.iter().map(|s| s.into_iter().last().unwrap()),
            ));
            seq.for_each_fon_seq(&mut Vec::new(), |s| {
                looking_for.remove(s);
            });
            TestResult::from_bool(looking_for.is_empty())
        }
    }
    QuickCheck::new()
        .gen(quickcheck::Gen::new(8))
        .quickcheck(fonset_seq_for_each_fon_seq_contains_samples as fn(_) -> TestResult);
}

#[quickcheck]
fn test_match_at_and_matches_at_same(seqs: (Vec<SmallFonSet>, Vec<SmallFonSet>)) -> TestResult {
    if seqs.0.is_empty() || seqs.1.is_empty() {
        return TestResult::discard();
    }
    let (pattern, word) = if seqs.0.len() <= seqs.1.len() {
        seqs
    } else {
        (seqs.1, seqs.0)
    };
    let pattern = unwrap_small_seq(pattern);
    let word = unwrap_small_seq(word);
    let mut result_buf = Vec::new();
    for word_idx in 0..(word.len() - pattern.len()) {
        let matched_at = pattern.match_at(&word, word_idx, &mut result_buf);
        assert_eq!(matched_at, pattern.matches_at(&word, word_idx));
        if matched_at {
            assert!(pattern.matches_at(&result_buf, word_idx));
        }
    }
    TestResult::passed()
}

#[test]
fn test_fonset_format_multiple() -> Result<()> {
    let mut reg = FonRegistry::new();
    let set = reg.set("blah")?;
    assert_eq!(format!("{}", set.format(&reg)), "[blah]");
    Ok(())
}

#[test]
fn test_fonset_format_single() -> Result<()> {
    let mut reg = FonRegistry::new();
    let set = reg.set("q")?;
    assert_eq!(format!("{}", set.format(&reg)), "q");
    Ok(())
}

#[test]
fn test_fonset_format_empty() -> Result<()> {
    let reg = FonRegistry::new();
    assert_eq!(format!("{}", FonSet::EMPTY.format(&reg)), "[]");
    Ok(())
}

#[test]
fn test_fonset_format() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        format!("{}", reg.setseq(&["z"])?.format_seq(&reg)),
        r#""z""#
    );
    assert_eq!(
        format!(
            "{}",
            reg.setseq(&["abc", "r", "def", "z"])?.format_seq(&reg)
        ),
        r#""[abc]r[def]z""#
    );
    Ok(())
}

#[test]
fn test_fonsetseq_format_empty() -> Result<()> {
    let reg = FonRegistry::new();
    assert_eq!(format!("{}", [].format_seq(&reg)), r#""""#);
    Ok(())
}
