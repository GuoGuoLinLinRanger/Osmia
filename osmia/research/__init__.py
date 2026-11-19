"""Research layer: hypotheses -> pre-registered replay tests -> baselines -> verdicts.

data       load a [time x asset] price matrix from market_history
replay     StrategySpec + a no-lookahead backtest of pair mean reversion
stats      fee-aware trade stats and day-block bootstrap confidence intervals
hypotheses LLM (or heuristic) hypothesis proposals, validated into StrategySpecs
journal    the research notebook
loop       one research cycle: baseline -> propose -> test -> decide -> write up
"""
