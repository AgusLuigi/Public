
1. Historienlänge (Data Availability)

1.1 Anzahl Perioden 

Name: **n_periods** (evt. neuer Name ;)
Formel:
n = Anzahl Zeitpunkte (An wievielen Tagen hatte der Store geöffnet in Tagen/Wochen)

Warum:
Ohne genügend Daten kann kein Modell lernen.


Typische Schwellen:

Ebene	min
Daily	180–365
Weekly	26–52
Monthly	12–24

⸻

2. Nachfragehäufigkeit (Demand Frequency)

2.1 Anzahl Verkaufsperioden

Name: **n_nonzero** (wie oft wurde ein Item in einem Store verkauft)
Formel:
n_{nz} = \sum_{t=1}^{n} \mathbb{1}(y_t > 0)

Warum:
Ein Modell braucht echte Ereignisse, nicht nur Nullen.

Typische Schwellen:

Ebene	min
Daily	20–40
Weekly	8–12


⸻

2.2 Zero Rate (Sparsity)

Name: **zero_rate**
Formel:
ZR = \frac{\sum_{t=1}^{n} \mathbb{1}(y_t = 0)}{n}

Warum:
Je höher, desto mehr Rauschen, desto weniger Signal.

Daumenwerte:

ZR	Bedeutung
< 0.6	normal
0.6–0.9	intermittent
> 0.9	meist unforecastable


⸻

3. Intermittency (Syntetos-Boylan)

Das ist der wichtigste professionelle Indikator.

3.1 ADI – Average Demand Interval

Name: **adi**
Formel:
ADI = \frac{n}{n_{nz}}

Interpretation:
Durchschnittliche Tage/Wochen zwischen Verkäufen.

ADI	Bedeutung
≈1	fast täglich
2–5	gelegentlich
>5	stark sporadisch
>10	quasi zufällig


⸻

4. Volatilität

4.1 Coefficient of Variation Squared

Name: **cv2**
Formel:
CV^2 = \left(\frac{\sigma}{\mu}\right)^2

mit
\mu = Mittelwert(y)
\sigma = Standardabweichung(y)

Warum:
Misst Chaos der Mengen, unabhängig von Skala.

CV²	Bedeutung
< 0.5	stabil
0.5–1.5	normal
> 2	hoch volatil
> 5	kaum lernbar


⸻

5. Klassische Forecastability-Matrix (Goldstandard)

Die Kombination aus ADI & CV² ist der bekannteste professionelle Klassifikator:

Typ		ADI		CV²

Smooth	≤ 1.32	≤ 0.49
Erratic	≤ 1.32	> 0.49
Intermittent	> 1.32	≤ 0.49
Lumpy	> 1.32	> 0.49

(Laut Syntetos & Boylan, 2005)

Interpretation:

Typ					Forecastbarkeit
Smooth				sehr gut
Erratic				gut
Intermittent		nur spezielle Modelle
Lumpy				oft nicht sinnvoll


⸻

6. Signal-Rausch-Verhältnis

6.1 SNR (Signal to Noise Ratio)

SNR = \frac{\text{Var(Trend + Seasonality)}}{\text{Var(Residuen)}}

In Praxis approximiert durch:
SNR \approx \frac{\mu}{\sigma}

Daumenwerte:

SNR	Bedeutung
> 1	brauchbares Signal
0.5–1	schwierig
< 0.5	fast nur Rauschen


⸻

7. Trend-Stärke (optional, aber mächtig)

7.1 Trend-R²

Regression:
y_t = a + b t + \epsilon_t

R^2 = 1 - \frac{\sum \epsilon_t^2}{\sum (y_t-\bar y)^2}

R²	Bedeutung
> 0.3	echter Trend
< 0.1	kein Trend


⸻

8. Seasonality Strength

8.1 Seasonal Strength (Hyndman)

S = 1 - \frac{Var(residuals)}{Var(residuals + seasonal)}

S	Bedeutung
> 0.3	klare Saisonalität
< 0.1	keine Struktur


⸻

Minimal-Set (das wirklich reicht)

Wenn du nur die wichtigsten 5 willst, nimm:

**Metrik**	**Muss**
**n_periods**	**ja**
**n_nonzero**	**ja**
**zero_rate**	**ja**
**ADI**	**ja**
**CV²**	**ja**

Damit kannst du schon 90 % aller schlechten Serien korrekt erkennen.

⸻

Konkrete Entscheidungsregel (Daily, praxistauglich)

forecastable = True
IF n_days < 180 → False
IF n_sales_days < 30 → False
IF zero_rate > 0.95 → False
IF ADI > 10 → False

und Klassifikation:

IF ADI > 1.5 AND CV² > 1.0 → intermittent
ELSE → regular


⸻

Warum genau diese Metriken?

Weil sie:
	1.	modellunabhängig sind
	2.	skaleninvariant sind
	3.	ex ante berechenbar sind (ohne Backtesting)
	4.	direkt physikalische Bedeutung haben:
		•	Zeit zwischen Events
		•	Rauschanteil
		•	Informationsdichte

⸻

Merksatz aus der Praxis

Forecastability ist kein ML-Problem.
Es ist ein Informationsdichte-Problem.

Und diese Metriken messen exakt diese Informationsdichte.





-----  **Active Span Days** ------ 

Lange Verkaufslücken sind oft Delisting oder OOS, nicht Intermittenz.

Best-Practice-Regel
	•	Definiere max_gap_days (typisch: 30–90 Öffnungstage, je nach Kategorie)
	•	Wenn zwischen zwei Verkaufstagen:

\Delta t > max\_gap\_days

→ Split in getrennte Active-Spans

Für ADI empfehle ich:
	•	Nur den letzten Active-Span verwenden
(repräsentiert aktuelle Nachfragecharakteristik)