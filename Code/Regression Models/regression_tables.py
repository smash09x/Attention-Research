"Builds Table 1, A5, A6, A6b, A7. A1-A4: model_main.py. A8: survival_models.py. Run: python regression_tables.py all"

import argparse

import numpy as np
import statsmodels.api as sm

from table_utils import (BASE_TERMS, FACTOR_TERMS, ERA_LABELS, add_category,
                         add_common_args, analytic, fit_ols, load_raw,
                         main_effect_term, platform_term)


def table_1(df):
    "Platform-driven effect per era (main effect + era x interaction), shown as exp(-b)."
    model = fit_ols(df, f"log_decay_speed ~ {BASE_TERMS}")
    names = list(model.params.index)
    pf = main_effect_term(model, 'platform_driven')
    inter = platform_term(model)

    print("\nTable 1")
    print("Difference in Attention Duration Between Platform-Driven and "
          "Non-Platform-Driven Content, by Era")
    print(f"{'Era':<10}{'b':>9}{'SE':>9}{'p':>9}{'ratio':>9}{'95% CI for ratio':>22}")

    for era in (0, 1, 2):
        contrast = np.zeros(len(names))
        contrast[names.index(pf)] = 1.0
        contrast[names.index(inter)] = era
        test = model.t_test(contrast)

        b = float(np.squeeze(test.effect))
        se = float(np.squeeze(test.sd))
        p = float(np.squeeze(test.pvalue))
        lo, hi = np.squeeze(test.conf_int())

        "Ratio is exp(-b), so the CI bounds swap."
        ratio, r_lo, r_hi = np.exp(-b), np.exp(-hi), np.exp(-lo)
        print(f"{ERA_LABELS[era]:<10}{b:>9.3f}{se:>9.3f}{p:>9.3f}"
              f"{ratio:>9.2f}{f'[{r_lo:.2f}, {r_hi:.2f}]':>22}")

    print(f"\nn = {int(model.nobs)}. HC3 standard errors.")


def table_a5(df):
    "Model A with era as an unordered factor, both DVs, plus the nested F test."
    print("\nTable A5")
    print("Model A Refitted With Era as an Unordered Factor, Primary Dataset")

    for outcome in ('log_decay_speed', 'log_decay_speed_post'):
        if outcome not in df.columns:
            continue
        sub = df.dropna(subset=[outcome])
        model = fit_ols(sub, f"{outcome} ~ {FACTOR_TERMS}")
        print(f"\n--- {outcome} (n = {int(model.nobs)}) ---")
        for name in model.params.index:
            print(f"{name:<55}{model.params[name]:>9.3f} ({model.bse[name]:.3f})"
                  f"  p = {model.pvalues[name]:.4f}")
        print(f"R2 = {model.rsquared:.3f}   adj R2 = {model.rsquared_adj:.3f}")

    "F test uses classical covariance, as an F test assumes."
    sub = df.dropna(subset=['log_decay_speed'])
    ordinal = fit_ols(sub, f"log_decay_speed ~ {BASE_TERMS}", robust=False)
    factor = fit_ols(sub, f"log_decay_speed ~ {FACTOR_TERMS}", robust=False)
    table = sm.stats.anova_lm(ordinal, factor)
    print(f"\nNested F test, ordinal vs factor: "
          f"F({int(table['df_diff'].iloc[-1])}, {int(factor.df_resid)}) = "
          f"{table['F'].iloc[-1]:.2f}, p = {table['Pr(>F)'].iloc[-1]:.3f}")


def table_a6(df):
    "Leave-one-category-out refits of the interaction, plus median peak widths."
    df = add_category(df)
    print("\nTable A6")
    print("Dependence of the Era x Platform-Driven Interaction on Each Category, "
          "and Median Peak Width by Era")
    print(f"{'Omitted':<16}{'n':>5}{'b':>9}{'p':>9}"
          f"{'2015-19':>10}{'2020-22':>10}{'2023-26':>10}{'cells':>14}")

    for category in ('control', 'movie', 'platform_event', 'viral'):
        sub = df[df['category'] != category]
        model = fit_ols(sub, f"log_decay_speed ~ {BASE_TERMS}")
        inter = platform_term(model)

        rows = df[df['category'] == category]
        medians = [rows.loc[rows['era_ordinal'] == e, 'peak_width_days'].median()
                   for e in (0, 1, 2)]
        counts = [int((rows['era_ordinal'] == e).sum()) for e in (0, 1, 2)]

        print(f"{category:<16}{int(model.nobs):>5}{model.params[inter]:>9.3f}"
              f"{model.pvalues[inter]:>9.3f}"
              + "".join(f"{m:>10.1f}" for m in medians)
              + f"{str(tuple(counts)):>14}")

    full = fit_ols(df, f"log_decay_speed ~ {BASE_TERMS}")
    inter = platform_term(full)
    print(f"{'None (full)':<16}{int(full.nobs):>5}{full.params[inter]:>9.3f}"
          f"{full.pvalues[inter]:>9.3f}")
    print("\nOmitting complementary categories gives identical estimates because the "
          "two dummies are a one-to-one recoding of the four categories.")


def table_a6b(df):
    "Era slope fitted separately for each category, with CIs. Answers which category accelerates."
    df = add_category(df)
    model = fit_ols(df, "log_decay_speed ~ C(category) * era_ordinal")
    names = list(model.params.index)

    def slope(category):
        contrast = np.zeros(len(names))
        contrast[names.index('era_ordinal')] = 1.0
        term = f"C(category)[T.{category}]:era_ordinal"
        if term in names:
            contrast[names.index(term)] = 1.0
        return contrast

    print("\nTable A6b")
    print("Era Slope Fitted Separately Within Each Content Category")
    print(f"{'Category':<16}{'n':>5}{'b':>9}{'SE':>8}{'p':>8}"
          f"{'95% CI':>20}{'ratio/era':>11}")

    for category in ('control', 'movie', 'platform_event', 'viral'):
        test = model.t_test(slope(category))
        b = float(np.squeeze(test.effect))
        se = float(np.squeeze(test.sd))
        p = float(np.squeeze(test.pvalue))
        lo, hi = np.squeeze(test.conf_int())
        n = int((df['category'] == category).sum())
        print(f"{category:<16}{n:>5}{b:>9.3f}{se:>8.3f}{p:>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>20}{np.exp(-b):>11.2f}")

    print("\nContrasts between category slopes:")
    for first, second in (('viral', 'movie'), ('platform_event', 'control'),
                          ('viral', 'control'), ('platform_event', 'movie')):
        test = model.t_test(slope(first) - slope(second))
        print(f"  {first} - {second:<16}{float(np.squeeze(test.effect)):>8.3f} "
              f"(SE {float(np.squeeze(test.sd)):.3f})  p = {float(np.squeeze(test.pvalue)):.3f}")

    print(f"\nn = {int(model.nobs)}. HC3 standard errors. The first two contrasts are the two "
          "distinct\nvalues in Table A6: the leave-one-out refits are two tests, not four.")


def table_a7(primary_raw, age_raw):
    "Rows excluded by censoring, by era and platform-driven status, both datasets."
    print("\nTable A7")
    print("Observations Excluded by Censoring, by Era and Platform-Driven Status")
    print(f"{'Dataset and era':<22}{'Obs pd=0':>10}{'Excl pd=0':>16}"
          f"{'Obs pd=1':>10}{'Excl pd=1':>16}{'Total excl':>14}")

    for label, data in (("Primary", primary_raw), ("Age proxy", age_raw)):
        if data is None:
            continue
        for era in (0, 1, 2):
            rows = data[data['era_ordinal'] == era]
            line = f"{label + ': ' + ERA_LABELS[era]:<22}"
            total_excluded = 0
            for flag in (0, 1):
                grp = rows[rows['platform_driven'].astype(int) == flag]
                excl = int(grp['censored'].sum())
                total_excluded += excl
                pct = (100 * excl / len(grp)) if len(grp) else 0.0
                line += f"{len(grp):>10}{f'{excl} ({pct:.0f}%)':>16}"
            print(line + f"{f'{total_excluded} of {len(rows)}':>14}")


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(
        description="Build Table 1, A5, A6, A6b and A7."))
    parser.add_argument('table', choices=['1', 'A5', 'A6', 'A6b', 'A7', 'all'])
    args = parser.parse_args()

    primary_raw = load_raw(args.trends, args.strat, args.youtube)
    primary = analytic(primary_raw)

    age_raw = None
    if args.table in ('A7', 'all'):
        age_raw = load_raw(args.trends_age, args.strat_age, args.youtube_age)

    if args.table in ('1', 'all'):
        table_1(primary)
    if args.table in ('A5', 'all'):
        table_a5(primary)
    if args.table in ('A6', 'all'):
        table_a6(primary)
    if args.table in ('A6b', 'all'):
        table_a6b(primary)
    if args.table in ('A7', 'all'):
        table_a7(primary_raw, age_raw)
