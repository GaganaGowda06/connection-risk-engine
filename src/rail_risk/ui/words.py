"""Plain-language phrasing of probabilities for the page."""


def chance_in_words(p: float) -> str:
    """A probability as a chance a person can picture: 'about 1 in 5', never a bare decimal."""
    if p <= 0.0:
        return "no chance"
    if p < 0.01:
        return "less than 1 in 100"
    if p >= 0.9:
        return "almost certain"
    if p >= 0.5:
        return f"more likely than not (about {p:.0%})"
    if p >= 0.35:  # 1 in 2 or 1 in 3 would misstate 0.4, so use tenths
        return f"about {round(p * 10)} in 10"
    return f"about 1 in {max(2, round(1 / p))}"


def miss_sentence(p: float, departure: str) -> str:
    """For example 'About 1 in 5 chance you miss the 10:06.'"""
    chance = chance_in_words(p)
    if chance == "no chance":
        return f"You are very unlikely to miss the {departure}."
    if chance == "almost certain":
        return f"You will almost certainly miss the {departure}."
    if chance.startswith("more likely"):
        return f"It is {chance} that you miss the {departure}."
    return f"{chance[0].upper()}{chance[1:]} chance you miss the {departure}."
