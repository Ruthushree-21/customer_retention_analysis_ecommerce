export_cols = demo_cols + score_cols + ["Retention_Class"]

df[export_cols].to_csv(
    "amazon_analysis_output.csv",
    index=False
)

print("\nSaved: amazon_analysis_output.csv")
print("Python analysis completed successfully!")