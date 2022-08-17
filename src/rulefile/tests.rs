use super::*;

#[test]
fn test_cost_ok() {
    assert_eq!(cost("12"), Ok(("", 12)));
    assert_eq!(cost("0"), Ok(("", 0)));
    assert_eq!(cost("92837498"), Ok(("", 92837498)));
    assert_eq!(cost("3"), Ok(("", 3)));
}

#[test]
fn test_cost_invalid_negative() {
    assert!(cost("-44").is_err());
    assert!(cost("-7").is_err());
    assert!(cost("-237819").is_err());
    assert!(cost("-0").is_err());
}

#[test]
fn test_cost_invalid() {
    assert!(cost("xyz").is_err());
    assert!(cost("twenty2").is_err());
}

#[test]
fn test_cost_invalid_fractional() {
    let mut cost = terminated(cost, eof);
    assert!(cost("1.0").is_err());
    assert!(cost("0.0").is_err());
    assert!(cost("1.1").is_err());
    assert!(cost("0.1").is_err());
    assert!(cost(".0").is_err());
    assert!(cost(".1").is_err());
    assert!(cost("-1.0").is_err());
    assert!(cost("-0.0").is_err());
    assert!(cost("-1.1").is_err());
    assert!(cost("-0.1").is_err());
    assert!(cost("-.0").is_err());
    assert!(cost("-.1").is_err());
}

#[test]
fn test_rule_alias_ok() {
    assert_eq!(
        rule("A = [a á â]"),
        Ok((
            "",
            Rule::Alias("A", vec![Item::Char('a'), Item::Char('á'), Item::Char('â')])
        ))
    );
    assert_eq!(
        rule("Xx = [_ C]"),
        Ok(("", Rule::Alias("Xx", vec![Item::None, Item::Alias("C")])))
    );
    assert_eq!(
        rule("Stuff = [x y Things]"),
        Ok((
            "",
            Rule::Alias(
                "Stuff",
                vec![Item::Char('x'), Item::Char('y'), Item::Alias("Things")]
            )
        ))
    );
}

#[test]
fn test_rule_alias_bad() {
    let mut rule = terminated(rule, eof);
    assert!(rule("lower = [a b]").is_err());
    assert!(rule("Upper = lower").is_err());
    assert!(rule("X = % @").is_err());
    assert!(rule("[x y] = Z").is_err());
    assert!(rule("C = [a b] [c d]").is_err());
}

#[test]
fn test_rule_norm() {
    assert_eq!(
        rule("x y > z"),
        Ok((
            "",
            Rule::Norm {
                from: vec![vec![Item::Char('x')], vec![Item::Char('y')]],
                to: vec![Item::Char('z')]
            }
        ))
    );
    assert_eq!(
        rule("x > y z"),
        Ok((
            "",
            Rule::Norm {
                from: vec![vec![Item::Char('x')]],
                to: vec![Item::Char('y'), Item::Char('z')]
            }
        ))
    );
}

#[test]
fn test_rule_mut() {
    assert_eq!(
        rule("3: [C x] | n > m | _"),
        Ok((
            "",
            Rule::Mut {
                cost: 3,
                before: vec![vec![Item::Alias("C"), Item::Char('x')]],
                from: vec![vec![Item::Char('n')]],
                to: vec![vec![Item::Char('m')]],
                after: vec![vec![Item::None]],
            }
        ))
    );
    assert_eq!(
        rule("0: [x] > [y z] N | a b c"),
        Ok((
            "",
            Rule::Mut {
                cost: 0,
                before: vec![],
                from: vec![vec![Item::Char('x')]],
                to: vec![
                    vec![Item::Char('y'), Item::Char('z')],
                    vec![Item::Alias("N")]
                ],
                after: vec![
                    vec![Item::Char('a')],
                    vec![Item::Char('b')],
                    vec![Item::Char('c')]
                ],
            }
        ))
    );
    assert_eq!(
        rule("20 : Blah | a b > [Z z]"),
        Ok((
            "",
            Rule::Mut {
                cost: 20,
                before: vec![vec![Item::Alias("Blah")]],
                from: vec![vec![Item::Char('a')], vec![Item::Char('b')]],
                to: vec![vec![Item::Alias("Z"), Item::Char('z')]],
                after: vec![],
            }
        ))
    );
    assert_eq!(
        rule("99: q > Q"),
        Ok((
            "",
            Rule::Mut {
                cost: 99,
                before: vec![],
                from: vec![vec![Item::Char('q')]],
                to: vec![vec![Item::Alias("Q")]],
                after: vec![],
            }
        ))
    );
}

#[test]
fn test_empty_line() {
    assert_eq!(rule_line(""), Ok(("", None)));
    assert_eq!(rule_line("   "), Ok(("", None)));
    assert_eq!(rule_line("; what ever ; "), Ok(("", None)));
    assert_eq!(rule_line("\t;stuff"), Ok(("", None)));
}
