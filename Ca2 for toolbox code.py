# 📦 Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 🎯 Set Plot Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ------------------------------------------------------------------------------
# 1. 📂 DATA LOADING & INSPECTION
# ------------------------------------------------------------------------------

# Load the IPL dataset
df = pd.read_csv(r"C:\Users\ashme\OneDrive\Desktop\4-Ashmeet\SEM 4\New folder\ipldataset.csv")

# View dataset info
print("📋 Dataset Info:")
print(df.info())

# ------------------------------------------------------------------------------
# 2. 📊 SUMMARY STATISTICS & NULL VALUES
# ------------------------------------------------------------------------------

# View basic statistics
print("\n📊 Statistical Summary:")
print(df.describe(include='all'))

# Check for missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())

# Column names
print("\n🧾 Column Names:")
print(df.columns.tolist())

# ------------------------------------------------------------------------------
# 3. 📊 BAR CHARTS: TOTAL RUNS & WICKETS PER TEAM
# ------------------------------------------------------------------------------

# Get last ball of each innings to calculate final score/wickets
last_ball_per_innings = df.loc[df.groupby(['mid', 'bat_team'])['overs'].idxmax()]

# Total runs and wickets by team
final_scores_by_team = last_ball_per_innings.groupby("bat_team")["total"].sum().sort_values(ascending=False)
final_wickets_by_team = last_ball_per_innings.groupby("bowl_team")["wickets"].sum().sort_values(ascending=False)

# Plot: Total Runs by Batting Team
plt.figure()
sns.barplot(y=final_scores_by_team.index, x=final_scores_by_team.values, palette="viridis")
plt.title("🏏 Total Runs Scored at End of Innings (Per Team)")
plt.xlabel("Total Runs")
plt.ylabel("Batting Team")
plt.tight_layout()
plt.show()

# Plot: Total Wickets by Bowling Team
plt.figure()
sns.barplot(y=final_wickets_by_team.index, x=final_wickets_by_team.values, palette="rocket")
plt.title("🎯 Total Wickets Taken at End of Innings (Per Team)")
plt.xlabel("Total Wickets")
plt.ylabel("Bowling Team")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4. 🥧 PIE CHARTS: TOP 5 BAT & BOWL TEAMS
# ------------------------------------------------------------------------------

# Top 5 teams
top5_batting = final_scores_by_team.head(5)
top5_bowling = final_wickets_by_team.head(5)

# Pie Chart: Batting Teams
plt.figure(figsize=(7, 7))
plt.pie(top5_batting, labels=top5_batting.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("🏏 Top 5 Batting Teams by Total Runs")
plt.axis('equal')
plt.show()

# Pie Chart: Bowling Teams
plt.figure(figsize=(7, 7))
plt.pie(top5_bowling, labels=top5_bowling.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
plt.title("🎯 Top 5 Bowling Teams by Total Wickets")
plt.axis('equal')
plt.show()

# ------------------------------------------------------------------------------
# 5. 📉 SCATTER & REGRESSION ANALYSIS: WICKETS VS TOTAL RUNS
# ------------------------------------------------------------------------------

# Scatter: Overs vs Total Runs
plt.figure()
plt.scatter(last_ball_per_innings["overs"], last_ball_per_innings["total"], color='darkcyan', alpha=0.6)
plt.title("Scatter Plot: Overs vs Total Runs")
plt.xlabel("Overs Played")
plt.ylabel("Total Runs Scored")
plt.tight_layout()
plt.show()

# Regression: Wickets vs Total Runs
X = last_ball_per_innings["wickets"].values.reshape(-1, 1)
y = last_ball_per_innings["total"].values

# Train linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot regression
plt.figure()
plt.scatter(X, y, color="green", alpha=0.6, label="Actual")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.title("Regression Line: Wickets vs Total Runs")
plt.xlabel("Wickets Lost")
plt.ylabel("Total Runs Scored")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 6. 📦 BOX PLOT: OUTLIERS IN TEAM TOTALS
# ------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.boxplot(x=last_ball_per_innings["bat_team"], y=last_ball_per_innings["total"], palette="Set3")
plt.title("📦 Team-wise Outlier Detection - Total Runs per Innings")
plt.xlabel("Batting Team")
plt.ylabel("Total Runs Scored")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 7. 🔥 HEATMAP: CORRELATION BETWEEN NUMERICAL FEATURES
# ------------------------------------------------------------------------------

# Select numeric columns for correlation
heatmap_features = df[["overs", "runs", "wickets", "runs_last_5", "wickets_last_5", "striker", "non-striker", "total"]]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_features.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("📊 Heatmap of Numerical Features (Correlation)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 8. 📉 MEAN SQUARED ERROR (MSE) FROM REGRESSION
# ------------------------------------------------------------------------------

mse = mean_squared_error(y, y_pred)
print(f"\n📉 Mean Squared Error (Wickets vs Total Runs): {mse:.2f}")
