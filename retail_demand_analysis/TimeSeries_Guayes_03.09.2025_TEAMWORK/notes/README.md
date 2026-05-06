# TODOs

 - z-score
 - outliers
 - clustering ai for others categories
 - weekly items sales (high/low)



Perfekt ✅
Dann ist jetzt **der spaßige Teil** dran: **Was kannst du PandasAI überhaupt fragen?**
Kurz: **Alles, was sich logisch aus deinen Daten ableiten lässt.**

Ich gebe dir **konkrete, sofort nutzbare Prompt-Beispiele**, geordnet nach Kategorien.
Du kannst sie **1:1 kopieren**.

---

## 📊 1️⃣ Daten verstehen (immer zuerst sinnvoll)

**Struktur & Überblick**

* `Explain this dataset in simple terms.`
* `What does each column represent?`
* `How many rows and columns are there?`
* `Which columns are numeric and which are categorical?`

**Qualität**

* `Are there missing values? Show me where.`
* `Which columns have outliers?`
* `Are there duplicate rows?`
* `Is this dataset clean enough for analysis?`

---

## 📈 2️⃣ Analysen & Statistiken

**Grundlegend**

* `Give me basic statistics for all numeric columns.`
* `What is the average, median, and standard deviation of column X?`
* `Which column has the highest variance?`

**Vergleiche**

* `Compare column A and column B.`
* `Which category has the highest average value of X?`
* `Show differences between groups in column Y.`

---

## 🔍 3️⃣ Muster, Trends & Insights

* `What patterns do you see in this data?`
* `Are there correlations between any variables?`
* `Which factors most influence column X?`
* `Is there any seasonality or trend in this dataset?`

Zeitreihen (falls Datum vorhanden):

* `How does X change over time?`
* `Are there unusual spikes or drops?`

---

## 🧪 4️⃣ Hypothesen & Fragen testen

* `Test whether column A significantly affects column B.`
* `Is there a meaningful difference between group A and group B?`
* `Does X predict Y? Explain why or why not.`

---

## 🧠 5️⃣ „Explain like I’m human“ (sehr stark)

* `Explain the main insights as if I were a non-technical stakeholder.`
* `Summarize the dataset in 5 bullet points.`
* `What would be the key takeaways for a manager?`

---

## 📉 6️⃣ Visualisierungen (automatisch generiert)

* `Plot the distribution of column X.`
* `Create a bar chart of X grouped by Y.`
* `Show a line chart of X over time.`
* `Visualize correlations between numeric columns.`

👉 PandasAI erzeugt den Code + das Ergebnis selbst.

---

## 🧩 7️⃣ Daten transformieren

* `Create a new column that is X divided by Y.`
* `Normalize column X.`
* `Filter rows where column A > 100.`
* `Group by column Y and compute the mean of X.`

---

## 🚨 8️⃣ Debugging & Datenprobleme finden

* `Why does column X contain negative values?`
* `Are there impossible or suspicious values?`
* `Check if any values violate logical constraints.`

---

## 🧠 9️⃣ Advanced / Power Prompts

* `If this were a machine learning dataset, what would be the target and features?`
* `How would you prepare this dataset for a regression model?`
* `What additional data would improve the analysis?`
* `What assumptions are we making with this data?`

---

## 🧪 🔥 10️⃣ Sehr guter Start-Prompt (empfohlen)

```text
First, explain the structure of the dataset.
Then identify data quality issues.
Finally, give me 3 interesting insights I should explore further.
```

---

## 🧠 Wichtiger Tipp

PandasAI **führt echten Python-Code aus**.
Das heißt:

* es rechnet wirklich
* es halluciniert weniger als ChatGPT
* schlechte Prompts → schlechte Analysen

👉 **Je klarer deine Frage, desto besser das Ergebnis.**
