use super::*;

use std::collections::BTreeSet;

use quickcheck::{Arbitrary, QuickCheck, TestResult};
use quickcheck_macros::*;

use rulefile::Item;

impl Arbitrary for Fon {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        Fon {
            id: u8::arbitrary(g) % MAX_FON_ID,
        }
    }
}

impl Arbitrary for FonSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        FonSet {
            bits: u128::arbitrary(g),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SmallFonSet(FonSet);

impl Arbitrary for SmallFonSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        SmallFonSet(FonSet {
            bits: u16::arbitrary(g) as u128,
        })
    }
}

#[derive(Debug, Clone)]
struct NonEmptyVec<T>(Vec<T>);

impl<T: Arbitrary> Arbitrary for NonEmptyVec<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        loop {
            let v = Vec::arbitrary(g);
            if !v.is_empty() {
                return NonEmptyVec(v);
            }
        }
    }
}

impl<T> AsRef<[T]> for NonEmptyVec<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T> IntoIterator for NonEmptyVec<T> {
    type Item = T;

    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
struct NonEmptyFonSetSeq(Vec<FonSet>);

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

#[derive(Debug, Clone)]
struct WithIndex<A: Arbitrary> {
    item: A,
    index: usize,
}

impl<A> Arbitrary for WithIndex<A>
where
    A: Arbitrary,
    A: IntoIterator,
    A: AsRef<[A::Item]>,
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        loop {
            let item = A::arbitrary(g);
            let len = item.as_ref().len();
            if len > 0 {
                let index = usize::arbitrary(g) % len;
                return WithIndex { item, index };
            }
        }
    }
}

#[derive(Debug, Clone)]
struct WithInsIndex<A: Arbitrary> {
    item: A,
    index: usize,
}

impl<A> Arbitrary for WithInsIndex<A>
where
    A: Arbitrary,
    A: IntoIterator,
    A: AsRef<[A::Item]>,
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let item = A::arbitrary(g);
        let index = usize::arbitrary(g) % (item.as_ref().len() + 1);
        WithInsIndex { item, index }
    }
}

#[test]
fn test_fon_registry_empty() {
    let reg = FonRegistry::new();
    for c in ['q', 'é', '\0'] {
        assert!(matches!(reg.get_fon(c), Err(NoSuchFon(c_)) if c_ == c));
    }
    for i in [1, 2, 35, MAX_FON_ID] {
        assert!(matches!(reg.get_fon_char(Fon::from(i)), Err(NoSuchFonId(i_)) if i_ == i));
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
fn test_fon_registry_add(chars: Vec<char>) -> Result<()> {
    let mut reg = FonRegistry::new();
    for c in chars {
        reg.add(c)?;
        assert_eq!(reg.get_fon_char(reg.get_fon(c)?)?, c);
    }
    Ok(())
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
    s |= Fon::from(101u8);
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
    assert!(FonSet::seq_is_empty(&setseq));
}

#[quickcheck]
fn test_fonsetseq_not_empty(setseq: NonEmptyFonSetSeq) -> bool {
    !FonSet::seq_is_empty(setseq.as_ref())
}

#[test]
fn test_fonsetseq_zero_len_empty() {
    assert!(FonSet::seq_is_empty(&[]));
}

#[test]
fn test_fonsetseq_real() {
    assert!(FonSet::seq_is_real(&[]));
    assert!(FonSet::seq_is_real(&[FonSet::EMPTY]));
    assert!(!FonSet::seq_is_real(&[FonSet::from(NO_FON)]));
    assert!(!FonSet::seq_is_real(&[
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::from(NO_FON),
        FonSet::from(Fon::from(80u8)),
    ]));
    assert!(FonSet::seq_is_real(&[
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::EMPTY,
        FonSet::from(Fon::from(80u8)),
    ]));
}

#[test]
fn test_fonsetseq_valid() {
    assert!(FonSet::seq_is_valid(&[]));
    assert!(FonSet::seq_is_valid(&[FonSet::EMPTY]));
    assert!(FonSet::seq_is_valid(&[FonSet::from(NO_FON)]));
    assert!(!FonSet::seq_is_valid(&[
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::from(NO_FON),
        FonSet::from(Fon::from(80u8)),
    ]));
    assert!(FonSet::seq_is_valid(&[
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::EMPTY,
        FonSet::from(Fon::from(80u8)),
    ]));
    assert!(FonSet::seq_is_valid(&[
        FonSet::from(NO_FON),
        FonSet::from([2u8.into(), 3u8.into()]),
        FonSet::from(Fon::from(80u8)),
        FonSet::from(NO_FON)
    ]));
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
    FonSet::seq_for_each_fon_seq(&reg.setseq(&["a"])?, |s| seqs.push(s.into()));
    assert_eq!(seqs, vec![reg.seq("a")?]);
    seqs.clear();
    FonSet::seq_for_each_fon_seq(&reg.setseq(&["x", "abc", "bx"])?, |s| seqs.push(s.into()));
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
            FonSet::seq_for_each_fon_seq(&seq, |_| {
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
            FonSet::seq_for_each_fon_seq(&seq, |s| {
                looking_for.remove(s);
            });
            TestResult::from_bool(looking_for.is_empty())
        }
    }
    QuickCheck::new()
        .gen(quickcheck::Gen::new(8))
        .quickcheck(fonset_seq_for_each_fon_seq_contains_samples as fn(_) -> TestResult);
}

fn collect_matches<M>(matcher: &M, word: &[M::Alph]) -> Vec<(Vec<M::Alph>, usize)>
where
    M: MutationRule,
    M::Alph: Clone,
{
    let mut results: Vec<(Vec<M::Alph>, usize)> = Vec::new();
    matcher.for_each_match(word, |result, index| results.push((result.clone(), index)));
    results
}

fn collect_matches_at<M>(matcher: &M, index: usize, word: &[M::Alph]) -> Vec<Vec<M::Alph>>
where
    M: MutationRule,
    M::Alph: Clone,
{
    let mut results: Vec<Vec<M::Alph>> = Vec::new();
    matcher.for_each_match_at(word, index, |result| results.push(result.clone()));
    results
}

fn check_match_result_reasonable(
    word: &[FonSet],
    pattern: &[FonSet],
    result_buf: &mut Vec<FonSet>,
    word_idx: usize,
) {
    assert_eq!(result_buf.len(), word.len());
    assert!(word_idx <= word.len() - pattern.len());
    assert_eq!(result_buf[0..word_idx], word[0..word_idx]);
    assert!(!FonSet::seq_is_empty(result_buf));
}

#[quickcheck]
fn test_fonsetseq_for_each_match_fuzz(word: NonEmptyFonSetSeq, pattern: NonEmptyFonSetSeq) {
    let word = word.as_ref();
    let pattern = pattern.as_ref();
    pattern.for_each_match(word, |result_buf, word_idx| {
        check_match_result_reasonable(word, pattern, result_buf, word_idx);
    });
}

#[quickcheck]
fn test_fonsetseq_for_each_match_at_fuzz(
    word: WithIndex<NonEmptyFonSetSeq>,
    pattern: NonEmptyFonSetSeq,
) {
    let word_idx = word.index;
    let word = word.item.as_ref();
    let pattern = pattern.as_ref();
    pattern.for_each_match_at(word, word_idx, |result_buf| {
        check_match_result_reasonable(word, pattern, result_buf, word_idx);
    });
}

#[test]
fn test_fon_set_seq_for_each_match_start() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["a"])?, &reg.setseq(&["ax", "bx"])?),
        vec![(reg.setseq(&["a", "bx"])?, 0),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_end() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["b"])?, &reg.setseq(&["ax", "bx"])?),
        vec![(reg.setseq(&["ax", "b"])?, 1),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_middle() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["b"])?, &reg.setseq(&["ax", "bx", "cx"])?),
        vec![(reg.setseq(&["ax", "b", "cx"])?, 1),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_multiple() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(
            &reg.setseq(&["ab", "c"])?,
            &reg.setseq(&["r", "a", "cab", "cx", "na"])?
        ),
        vec![
            (reg.setseq(&["r", "a", "c", "cx", "na"])?, 1),
            (reg.setseq(&["r", "a", "ab", "c", "na"])?, 2),
        ]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_no_match() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["n"])?, &reg.setseq(&["a", "b"])?),
        vec![]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_longer_pattern() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["a", "b"])?),
        vec![]
    );
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["b", "c"])?),
        vec![]
    );
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["a"])?),
        vec![]
    );
    Ok(())
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
fn test_registry_normalize_lowercases() -> Result<()> {
    let mut reg = FonRegistry::new();
    for c in "great".chars() {
        reg.add(c)?;
    }
    assert_eq!(reg.normalize("GrEat")?, reg.normalize("great")?);
    Ok(())
}

#[test]
fn test_dictionary_one_key() -> Result<()> {
    let normalized = &[Fon::from(8u8), Fon::from(2u8)];
    let mut cfg = BuscaCfg::new();
    assert_eq!(Vec::from_iter(cfg.words_iter(normalized)), &[] as &[&str]);

    cfg.add_to_dictionary("first", normalized)?;
    assert_eq!(Vec::from_iter(cfg.words_iter(normalized)), &["first"]);

    cfg.add_to_dictionary("another", normalized)?;
    assert_eq!(
        BTreeSet::from_iter(cfg.words_iter(normalized)),
        BTreeSet::from(["another", "first"])
    );
    Ok(())
}

#[test]
fn test_dictionary_two_keys() -> Result<()> {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)];
    let mut cfg = BuscaCfg::new();
    cfg.add_to_dictionary("first", norm1)?;
    cfg.add_to_dictionary("another", norm1)?;
    assert_eq!(Vec::from_iter(cfg.words_iter(norm2)), &[] as &[&str]);

    cfg.add_to_dictionary("word", norm2)?;
    assert_eq!(Vec::from_iter(cfg.words_iter(norm2)), &["word"]);

    assert_eq!(
        BTreeSet::from_iter(cfg.words_iter(norm1)),
        BTreeSet::from(["another", "first"])
    );
    Ok(())
}

#[test]
fn test_dictionary_duplicate_one_key() -> Result<()> {
    let key = &[Fon::from(33u8)];
    let mut cfg = BuscaCfg::new();
    assert_eq!(Vec::from_iter(cfg.words_iter(key)), &[] as &[&str]);

    cfg.add_to_dictionary("value", key)?;
    assert_eq!(Vec::from_iter(cfg.words_iter(key)), &["value"]);
    cfg.add_to_dictionary("value", key)?;
    assert_eq!(Vec::from_iter(cfg.words_iter(key)), &["value"]);
    Ok(())
}

#[test]
fn test_dictionary_duplicate_two_keys() -> Result<()> {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)];
    let mut cfg = BuscaCfg::new();
    cfg.add_to_dictionary("same", norm1)?;
    cfg.add_to_dictionary("same", norm2)?;
    assert_eq!(Vec::from_iter(cfg.words_iter(norm1)), &["same"]);
    assert_eq!(Vec::from_iter(cfg.words_iter(norm2)), &["same"]);
    Ok(())
}

#[test]
fn test_empty_ruleset() {
    let ruleset = NormalizeRuleSet::new();
    assert_eq!(ruleset.get_rule(&['a']), None);
    assert_eq!(ruleset.get_rule(&['b', 'c']), None);
    assert_eq!(ruleset.find_rule(&['d']), None);
    assert_eq!(ruleset.find_rule(&['e', 'f']), None);
}

#[test]
fn test_empty_ruleset_removes_end_anchors() {
    let ruleset = NormalizeRuleSet::new();
    assert_eq!(ruleset.get_rule(&[NO_FON_CHAR]), Some(&[] as &[_]));
}

#[test]
fn test_ruleset_add_get() -> Result<()> {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)] as &[_];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)] as &[_];
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['c'], norm1)?;
    assert_eq!(ruleset.longest_rule(), 1);
    ruleset.add_rule(&['a', 'b'], norm2)?;
    assert_eq!(ruleset.longest_rule(), 2);
    assert_eq!(ruleset.get_rule(&['a']), None);
    assert_eq!(ruleset.get_rule(&['a', 'b']), Some(norm2));
    assert_eq!(ruleset.get_rule(&['c']), Some(norm1));
    assert_eq!(ruleset.get_rule(&['x', 'c']), None);
    assert_eq!(ruleset.get_rule(&['c', 'x']), None);
    assert_eq!(ruleset.get_rule(&['d']), None);
    Ok(())
}

#[test]
fn test_ruleset_add_duplicate() -> Result<()> {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)];
    let norm3 = &[Fon::from(10u8)];
    let norm4 = &[Fon::from(100u8)];
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['c'], norm1)?;
    assert!(matches!(
        ruleset.add_rule(&['c'], norm2),
        Err(DuplicateNormRule(_))
    ));
    ruleset.add_rule(&['a', 'b'], norm3)?;
    assert!(matches!(
        ruleset.add_rule(&['c'], norm2),
        Err(DuplicateNormRule(_))
    ));
    assert!(matches!(
        ruleset.add_rule(&['a', 'b'], norm4),
        Err(DuplicateNormRule(_))
    ));
    Ok(())
}

#[test]
fn test_ruleset_find() -> Result<()> {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)] as &[_];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)] as &[_];
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['a', 'b'], norm1)?;
    ruleset.add_rule(&['c'], norm2)?;
    assert_eq!(ruleset.find_rule(&['x']), None);
    assert_eq!(ruleset.find_rule(&['x', 'a', 'b', 'y']), None);
    assert_eq!(ruleset.find_rule(&['x', 'c', 'y']), None);
    assert_eq!(ruleset.find_rule(&['c']), Some((1, norm2)));
    assert_eq!(ruleset.find_rule(&['c', 'z']), Some((1, norm2)));
    assert_eq!(ruleset.find_rule(&['a', 'b']), Some((2, norm1)));
    assert_eq!(ruleset.find_rule(&['a', 'b', 'n']), Some((2, norm1)));
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "mno".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['a', 'b', 'c'], &[reg.add('z')?])?;
    assert_eq!(
        (&ruleset, &reg).normalize("noabcm")?,
        reg.normalize("nozm")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_empty_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "normal".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['a', 'b', 'c'], &[])?;
    ruleset.add_rule(&['e'], &[])?;
    assert_eq!(
        (&ruleset, &reg).normalize("noabcrmael")?,
        reg.normalize("normal")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_lowercases() -> Result<()> {
    let ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "great".chars() {
        reg.add(c)?;
    }
    assert_eq!(
        (&ruleset, &reg).normalize("GrEat")?,
        (&ruleset, &reg).normalize("great")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_end_anchored_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "be".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['s', 't', '_'], &[reg.add('d')?])?;
    assert_eq!((&ruleset, &reg).normalize("best")?, reg.normalize("bed")?);
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_start_anchored_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "crow".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['_', 'w'], &[reg.add('c')?, reg.add('r')?])?;
    assert_eq!((&ruleset, &reg).normalize("wow")?, reg.normalize("crow")?);
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_empty_result_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "tests".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['s'], &[])?;
    assert_eq!((&ruleset, &reg).normalize("tests")?, reg.normalize("tet")?);
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_drop_from_start() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "worddrop".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['_', 'd', 'r', 'o', 'p'], &[])?;
    assert_eq!(
        (&ruleset, &reg).normalize("dropword")?,
        reg.normalize("word")?
    );
    assert_eq!(
        (&ruleset, &reg).normalize("worddrop")?,
        reg.normalize("worddrop")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_drop_from_end() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "worddrop".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['d', 'r', 'o', 'p', '_'], &[])?;
    assert_eq!(
        (&ruleset, &reg).normalize("worddrop")?,
        reg.normalize("word")?
    );
    assert_eq!(
        (&ruleset, &reg).normalize("dropword")?,
        reg.normalize("dropword")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_refuses_no_fon_char() -> Result<()> {
    let ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "abcjklm".chars() {
        reg.add(c)?;
    }
    assert!(matches!((&ruleset, &reg).normalize("_"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("_ab"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("c_"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("jk_lm"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("__"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("___"), Err(NoSuchFon(c)) if c == '_'));
    Ok(())
}

#[test]
fn test_buscacfg_normalize() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    for c in "mno".chars() {
        cfg.fon_registry.add(c)?;
    }
    cfg.normalize_rules
        .add_rule(&['a', 'b', 'c'], &[cfg.fon_registry.add('z')?])?;
    assert_eq!(
        cfg.normalize("noabcm")?,
        cfg.fon_registry.normalize("nozm")?
    );
    Ok(())
}

#[test]
fn test_buscacfg_search_word_in_dictionary_comes_first() -> Result<()> {
    let word = "word";
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("10: x > y".as_bytes())?;
    for c in word.chars() {
        cfg.fon_registry.add(c)?;
    }
    cfg.add_to_dictionary(word, &cfg.normalize(word)?)?;
    assert_eq!(cfg.search(word)?.flatten().next(), Some((word, 0)));
    Ok(())
}

#[test]
fn test_buscacfg_search_normalize_error() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("10: x > y".as_bytes())?;
    assert!(matches!(cfg.search("anything"), Err(NoSuchFon(_))));
    Ok(())
}

#[quickcheck]
fn test_set_alias(fons: FonSet) -> Result<()> {
    let mut cfg = BuscaCfg::new();
    assert_eq!(cfg.try_get_alias("blah"), None);
    assert!(matches!(cfg.get_alias("blah"), Err(NoSuchAlias(_))));
    cfg.set_alias("blah".into(), fons);
    assert_eq!(cfg.try_get_alias("blah"), Some(fons));
    assert_eq!(cfg.get_alias("blah")?, fons);
    assert_eq!(cfg.try_get_alias("blah!"), None);
    assert!(matches!(cfg.get_alias("blah!"), Err(NoSuchAlias(_))));
    Ok(())
}

#[test]
fn test_add_rule_alias() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Alias("Hi", vec![Item::Char('a')]))?;
    assert_eq!(
        cfg.get_alias("Hi")?,
        FonSet::from(cfg.fon_registry.get_fon('a')?)
    );
    Ok(())
}

#[test]
fn test_add_rule_alias_with_anchor() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Alias("Xy", vec![Item::Char('n'), Item::None]))?;
    assert_eq!(
        cfg.get_alias("Xy")?,
        FonSet::from([cfg.fon_registry.get_fon('n')?, NO_FON])
    );
    Ok(())
}

#[test]
fn test_resolve_norm_rule_result() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let r = cfg.resolve_norm_rule_result(&vec![Item::Char('o'), Item::Char('k')])?;
    assert_eq!(r, cfg.normalize("ok")?);
    Ok(())
}

#[test]
fn test_invalid_normalize_rule_rhs() {
    assert!(matches!(
        BuscaCfg::new().load_rules("a b > a _".as_bytes()),
        Err(InvalidNormRule(_))
    ));
    assert!(matches!(
        BuscaCfg::new().load_rules("a b > _ c".as_bytes()),
        Err(InvalidNormRule(_))
    ));
    assert!(matches!(
        BuscaCfg::new().load_rules("C = [z r]\nv > C".as_bytes()),
        Err(InvalidNormRule(_))
    ));
}

#[test]
fn test_invalid_normalize_rule_lhs() {
    assert!(BuscaCfg::new().load_rules(" > a".as_bytes()).is_err());
    assert!(BuscaCfg::new().load_rules("a _ b > c".as_bytes()).is_err());
    assert!(BuscaCfg::new().load_rules("_ _ _ > c".as_bytes()).is_err());
}

#[test]
fn test_resolve_rule_item_set_alias() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Alias("N")])?;
    assert_eq!(r, s);
    Ok(())
}

#[test]
fn test_resolve_rule_item_set_alias_plus() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Char('q'), Item::Alias("N")])?;
    assert_eq!(r, s | cfg.fon_registry.get_fon('q')?);
    Ok(())
}

#[test]
fn test_resolve_rule_item_set_alias_plus_anchor() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Alias("N"), Item::None])?;
    assert_eq!(r, s | NO_FON);
    Ok(())
}

#[test]
fn test_add_rule_norm_simple() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('a')], vec![Item::Char('b')]],
        to: vec![Item::Char('c'), Item::Char('d')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['a', 'b']),
        Some(cfg.normalize("cd")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_anchored_start() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::None], vec![Item::Char('x')]],
        to: vec![Item::Char('y')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&[NO_FON_CHAR, 'x']),
        Some(cfg.normalize("y")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_anchored_end() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('e')], vec![Item::None]],
        to: vec![Item::Char('r')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['e', NO_FON_CHAR]),
        Some(cfg.normalize("r")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_with_sets() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![
            vec![Item::Char('n')],
            vec![Item::Char('a'), Item::Char('o')],
        ],
        to: vec![Item::Char('y')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['n', 'o']),
        Some(cfg.normalize("y")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_adds_rhs_only_to_registry() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('a')], vec![Item::Char('b')]],
        to: vec![Item::Char('c')],
    })?;
    assert!(cfg.fon_registry.get_fon('c').is_ok());
    assert!(cfg.fon_registry.get_fon('a').is_err());
    assert!(cfg.fon_registry.get_fon('b').is_err());
    Ok(())
}

fn for_cartesian_product_results<T, S>(items: &[S]) -> Result<Vec<Vec<T>>>
where
    S: AsRef<[T]>,
    T: Copy + Default,
{
    let mut results = Vec::new();
    for_cartesian_product(items, |i| {
        results.push(Vec::from(i));
        Ok(())
    })?;
    Ok(results)
}

#[test]
fn test_for_cartesian_product() -> Result<()> {
    assert_eq!(
        for_cartesian_product_results(&[vec![3, 4], vec![7], vec![0, 10, 100]])?,
        vec![
            vec![3, 7, 0],
            vec![3, 7, 10],
            vec![3, 7, 100],
            vec![4, 7, 0],
            vec![4, 7, 10],
            vec![4, 7, 100],
        ]
    );
    Ok(())
}

#[test]
fn test_for_cartesian_product_len() {
    fn for_cartesian_product_len_is_product(items: Vec<Vec<u8>>) -> TestResult {
        let expected_len: usize = items.iter().map(Vec::len).product();
        if items.is_empty() || expected_len == 0 || expected_len > 10_000 {
            TestResult::discard()
        } else {
            let mut count: usize = 0;
            for_cartesian_product(&items, |_| {
                count += 1;
                Ok(())
            })
            .unwrap();
            TestResult::from_bool(count == expected_len)
        }
    }
    QuickCheck::new()
        .gen(quickcheck::Gen::new(8))
        .quickcheck(for_cartesian_product_len_is_product as fn(_) -> TestResult);
}

#[test]
fn test_for_cartesian_product_contains_samples() {
    fn for_cartesian_product_contains_samples(items: Vec<Vec<u8>>) -> TestResult {
        let expected_len: usize = items.iter().map(Vec::len).product();
        if items.is_empty() || expected_len == 0 || expected_len > 10_000 {
            TestResult::discard()
        } else {
            let mut looking_for = BTreeSet::new();
            looking_for.insert(Vec::from_iter(items.iter().map(|v| v[0])));
            looking_for.insert(Vec::from_iter(
                items.iter().map(|v| v.last().cloned().unwrap()),
            ));
            looking_for.insert(Vec::from_iter(
                items.iter().map(|v| v.iter().min().cloned().unwrap()),
            ));
            for_cartesian_product(&items, |x| {
                looking_for.remove(x);
                Ok(())
            })
            .unwrap();
            TestResult::from_bool(looking_for.is_empty())
        }
    }
    QuickCheck::new()
        .gen(quickcheck::Gen::new(8))
        .quickcheck(for_cartesian_product_contains_samples as fn(_) -> TestResult);
}

trait TestingFonRegistry {
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

#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct MockMutationRule<T> {
    matches: Vec<(Vec<T>, usize)>,
    matches_at: Vec<Vec<T>>,
}

impl<T: Clone> MutationRule for MockMutationRule<T> {
    type Alph = T;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        _word: &[Self::Alph],
        _word_idx: usize,
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for result in &self.matches_at {
            result_buf.clear();
            result_buf.extend_from_slice(result);
            action(result_buf);
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        _word: &[Self::Alph],
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for (result, idx) in &self.matches {
            result_buf.clear();
            result_buf.extend_from_slice(result);
            action(result_buf, *idx);
        }
    }
}

#[derive(Clone, Debug)]
struct ArbReplaceRule<T> {
    remove_idx: usize,
    remove_len: usize,
    replace_with: Vec<T>,
}

impl<T: Arbitrary> Arbitrary for ArbReplaceRule<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        ArbReplaceRule {
            remove_idx: u8::arbitrary(g) as usize,
            remove_len: u8::arbitrary(g) as usize,
            replace_with: Vec::arbitrary(g),
        }
    }
}

fn prep_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<ReplaceRule<MockMutationRule<T>>> {
    let rule = ReplaceRule {
        matcher: MockMutationRule {
            matches: vec![(match_val.clone(), match_idx)],
            matches_at: vec![match_val.clone()],
        },
        remove_idx: repl_args.remove_idx,
        remove_len: repl_args.remove_len,
        replace_with: repl_args.replace_with.into(),
    };
    if match_idx + rule.get_remove_end() > match_val.len() {
        None
    } else {
        Some(rule)
    }
}

fn match_mock_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<Vec<T>> {
    prep_replace_rule(match_val, match_idx, repl_args).map(|rule| {
        let (result, _) = collect_matches(&rule, &[]).into_iter().next().unwrap();
        result
    })
}

fn match_at_mock_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<Vec<T>> {
    prep_replace_rule(match_val, match_idx, repl_args).map(|rule| {
        collect_matches_at(&rule, match_idx, &[])
            .into_iter()
            .next()
            .unwrap()
    })
}

#[quickcheck]
fn replace_rule_for_each_match_inserts(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_mock_replace_rule(match_val, match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx..].starts_with(&repl_args.replace_with),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_at_inserts(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_at_mock_replace_rule(match_val, match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx..].starts_with(&repl_args.replace_with),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_removes(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_mock_replace_rule(match_val.clone(), match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx + repl_args.replace_with.len()..]
                .starts_with(&match_val[match_idx + repl_args.remove_idx + repl_args.remove_len..]),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_at_removes(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) =
        match_at_mock_replace_rule(match_val.clone(), match_idx, repl_args.clone())
    {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx + repl_args.replace_with.len()..]
                .starts_with(&match_val[match_idx + repl_args.remove_idx + repl_args.remove_len..]),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn test_replace_rule_set_for_each_match(matches: Vec<(Vec<u8>, usize)>) -> bool {
    let rule = MockMutationRule {
        matches,
        ..Default::default()
    };
    collect_matches(&ReplaceRuleSet::from(rule.clone()), &[]) == collect_matches(&rule, &[])
}

#[quickcheck]
fn test_replace_rule_set_for_each_match_at(matches_at: Vec<Vec<u8>>) -> bool {
    let rule = MockMutationRule {
        matches_at,
        ..Default::default()
    };
    collect_matches_at(&ReplaceRuleSet::from(rule.clone()), 0, &[])
        == collect_matches_at(&rule, 0, &[])
}

#[test]
fn test_replace_rule_cost_set_add() -> Result<()> {
    let rule: MockMutationRule<i32> = Default::default();
    let mut rule_set = ReplaceRuleCostSet::new();
    let cost: Cost = 3;
    rule_set.add_rule(rule.clone(), cost);
    let idx = rule_set.costs.iter().position(|&c| c == cost).unwrap();
    assert!(rule_set.rules[idx].rules.contains(&rule));
    Ok(())
}

#[quickcheck]
fn test_start_anchored_fonsetseq_rule(word: NonEmptyFonSetSeq, pattern: NonEmptyFonSetSeq) {
    let word = word.as_ref();
    let pattern = pattern.as_ref();
    let rule = StartAnchoredRule(pattern);
    assert_eq!(
        collect_matches(&rule, word),
        Vec::from_iter(
            collect_matches_at(&pattern, 0, word)
                .into_iter()
                .map(|r| (r, 0))
        )
    );
}

#[quickcheck]
fn test_end_anchored_fonsetseq_rule(word: NonEmptyFonSetSeq, pattern: NonEmptyFonSetSeq) {
    let word = word.as_ref();
    let pattern = pattern.as_ref();
    let rule = EndAnchoredRule(pattern);
    if pattern.len() > word.len() {
        assert_eq!(collect_matches(&rule, word), vec![]);
    } else {
        let index = word.len() - pattern.len();
        assert_eq!(
            collect_matches(&rule, word),
            Vec::from_iter(
                collect_matches_at(&pattern, index, word)
                    .into_iter()
                    .map(|r| (r, index))
            )
        );
    }
}

#[test]
fn test_both_anchored_fonsetseq_rule_same_len() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(
            &BothAnchoredRule(reg.setseq(&["ab", "cd", "ef"])?),
            &reg.setseq(&["b", "c", "e"])?
        ),
        vec![(reg.setseq(&["b", "c", "e"])?, 0)]
    );
    assert_eq!(
        collect_matches(
            &BothAnchoredRule(reg.setseq(&["ab", "cd"])?),
            &reg.setseq(&["a", "b"])?
        ),
        vec![]
    );
    Ok(())
}

#[test]
fn test_both_anchored_fonsetseq_rule_longer_rule() {
    let f = FonSet::from(Fon::from(5u8));
    assert_eq!(collect_matches(&BothAnchoredRule(vec![f, f]), &[f]), vec![]);
    assert_eq!(
        collect_matches(&BothAnchoredRule(vec![f, f, f]), &[f, f]),
        vec![]
    );
}

#[test]
fn test_both_anchored_fonsetseq_rule_longer_word() {
    let f = FonSet::from(Fon::from(5u8));
    assert_eq!(collect_matches(&BothAnchoredRule(vec![f]), &[f, f]), vec![]);
    assert_eq!(
        collect_matches(&BothAnchoredRule(vec![f, f]), &[f, f, f]),
        vec![]
    );
}

#[test]
fn test_sliceset_with_fon_set_seq() -> Result<()> {
    let mut set: BTreeSet<Box<[FonSet]>> = BTreeSet::new();
    let mut reg = FonRegistry::new();
    let setseq1 = reg.setseq(&["xy", "z"])?;
    let setseq2 = reg.setseq(&["z", "ay"])?;
    assert!(!set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq2);
    assert!(set.has_slice(&setseq1));
    assert!(set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(set.has_slice(&setseq2));
    Ok(())
}

#[test]
fn test_busca_node_inc_cost() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("0: a > b".as_bytes())?;
    cfg.load_rules("12: a > b".as_bytes())?;
    cfg.load_rules("3: a > b".as_bytes())?;
    cfg.load_rules("55: a > b".as_bytes())?;
    let word = cfg.fon_registry.setseq(&["a"])?;
    let mut node = BuscaNode::new(&cfg, &word, 0);
    assert_eq!(node.total_cost, 0);
    assert_eq!(node.inc_cost(&cfg), Some(3));
    assert_eq!(node.inc_cost(&cfg), Some(12));
    assert_eq!(node.inc_cost(&cfg), Some(55));
    assert_eq!(node.inc_cost(&cfg), None);
    Ok(())
}

#[test]
fn test_busca_node_cmp() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("0: a > b".as_bytes())?;
    let word = cfg.fon_registry.setseq(&["a"])?;
    let node1 = BuscaNode::new(&cfg, &word, 0);
    let node2 = BuscaNode::new(&cfg, &word, 78);
    assert!(node1 > node2);
    assert!(node2 < node1);
    assert!(node1 >= node1);
    assert!(node1 <= node1);
    assert!(node2 >= node2);
    assert!(node2 <= node2);
    Ok(())
}
