from clap.const import MULTI_SIMLEX_PATH
from clap.multi_simlex_utils import load_multi_simlex_raw, multi_simlex_to_df

if __name__ == "__main__":
    df = load_multi_simlex_raw()
    df = multi_simlex_to_df(df)
    df.to_csv(MULTI_SIMLEX_PATH.replace(".csv", "_processed.csv"), index=False)
    print(
        f"Processed dataset saved to {MULTI_SIMLEX_PATH.replace('.csv', '_processed.csv')}"
    )
    print("Multi-SimLex dataset processing completed.")
