# Soorten testen

## Central Limit Theorem (CLT) en gerelateerde tests

Het Centraal Limiet Theorema (CLT) stelt dat de som van een groot aantal onafhankelijke willekeurige variabelen ongeveer normaal verdeeld is. Hoe groter de steekproef, hoe beter de benadering. Bij de CLT is de standaardafwijking voor de steekproefverdeling gelijk aan de standaardafwijking van de populatie gedeeld door de wortel van de steekproefgrootte (sigma / sqrt(n)). De normale verdeling speelt hierbij een belangrijke rol.

### Z-test

Gebruik de z-test als aan de volgende vereisten voldaan is:

- Je hebt een willekeurige steekproef.
- De steekproef is groot genoeg (n >= 30). Als de populatie al normaal verdeeld is, is de steekproefgrootte niet relevant voor deze voorwaarde.
- De data is normaal verdeeld.
- De populatie standaarddeviatie is gekend. Indien één van deze voorwaarden niet voldaan is en de data wel normaal verdeeld is, gebruik je in plaats daarvan de t-test. De z-test kan worden uitgevoerd als een right-tailed, left-tailed, of two-tailed test, waarbij de stappen voor het formuleren van hypotheses, kiezen van significantieniveau, bepalen van de teststatistiek en $p$-waarde, en het trekken van conclusies worden gevolgd.

### T-test

De t-test wordt gebruikt als één van de vereisten voor de z-test niet voldaan is (behalve de normaliteitsvereiste). Dit omvat situaties met kleine steekproeven of wanneer de populatie standaarddeviatie onbekend is. De t-test maakt gebruik van de Student $t$-distributie. Net als de z-test kan de t-test right-tailed, left-tailed of two-tailed zijn. Confidence intervals voor kleine steekproeven gebruiken de Student t-test.

#### Confidence Intervals

Een confidence interval geeft een interval aan waarin de parameter met een bepaalde kans ligt. Voor grote steekproeven worden andere methoden gebruikt dan voor kleine steekproeven (Student t test).

## Bivariate - Kwalitatieve variabelen (Analyse van 2 kwalitatieve variabelen)

Dit betreft de analyse van associaties tussen twee kwalitatieve variabelen.

### Chi-kwadraattoets

Deze test wordt gebruikt om associaties tussen categorische variabelen te beoordelen.

- De nulhypothese ($H_0$) stelt dat er geen verband is tussen de 2 variabelen (de verschillen tussen geobserveerde en verwachte waarden zijn klein), terwijl de alternatieve hypothese ($H_1$) stelt dat er een verband is (de verschillen zijn groot).

- De testprocedure omvat het formuleren van hypotheses, kiezen van $\alpha$, berekenen van de $\chi^2$ teststatistiek (gebaseerd op $df = (r-1) \times (k-1)$), en het trekken van conclusies op basis van de kritieke waarde of $p$-waarde.
- De Chi-kwadraattoets geeft enkel juiste resultaten als er voldoende data is. Cochran's rule geeft hiervoor richtlijnen, zoals dat bij een 2x2 contingentietabel alle verwachte waarden > 1 moeten zijn en minstens 20% van de verwachte waarden > 5 moeten zijn.

#### Cramér's V

Meet de sterkte van de associatie tussen categorische variabelen. Het is een formule die de $\chi^2$ waarde normaliseert naar een waarde tussen 0 en 1 die onafhankelijk is van de tabelgrootte. Interpretaties variëren van 0 (geen associatie) tot 1 (complete associatie), met waarden als 0.1 (zwak), 0.25 (matig), 0.50 (sterk), 0.75 (zeer sterk).

### Goodness-of-fit test

Deze test controleert of de waargenomen frequenties overeenkomen met de verwachte theoretische verdeling.

- De nulhypothese ($H_0$) stelt dat de sample representatief is voor de populatie, d.w.z. de frequentie van elke klasse in de sample komt goed overeen met die in de populatie. De alternatieve hypothese ($H_1$) stelt dat de sample niet representatief is, d.w.z. de verschillen met de verwachte frequenties zijn te groot.
- De testprocedure lijkt op die van de Chi-kwadraattoets voor associatie, maar de degrees of freedom ($df$) zijn hier $(k-1)$, waarbij $k$ het aantal categorieën in de sample is.

### Standardised residuals

Deze worden gebruikt om te kijken of een sample overrepresentatief is voor een bepaalde groep, vaak na een Chi-kwadraattoets.

## Bivariate - Kwalitatief en Kwantitatief (Analyse van 1 kwalitatieve en 1 kwantitatieve variabele)

Dit betreft het vergelijken van een kwantitatieve variabele tussen groepen gedefinieerd door een kwalitatieve variabele.

### Independent t-test (two-sample t-test)

Gebruik deze test wanneer je de gemiddelden van twee onafhankelijke groepen of condities vergelijkt.

- Dit is van toepassing bij 2 onafhankelijke groepen.
- Het doel is het vergelijken van het gemiddelde van 2 groepen die niet perse even groot hoeven te zijn.
- Een voorbeeld is het vergelijken van het gemiddelde tussen een groep met een placebo en een groep met medicijn.

### Paired t-test (two-sample t-test)

Gebruik deze test bij het analyseren van gepaarde of gematchte observaties om gemiddelden te vergelijken binnen dezelfde groep onder verschillende condities of tijdstippen.

- Dit gaat over 2 afhankelijke groepen.
- Hierbij worden dezelfde testsubjecten gebruikt.
- Een voorbeeld is het testen van dezelfde auto met verschillende soorten benzine.

#### Cohen's d

Een maat voor de effectgrootte. Gebruik Cohen's d om de praktische significantie van het waargenomen verschil te interpreteren en effectgroottes te vergelijken tussen studies of condities. Het geeft aan hoe groot het verschil is tussen de 2 groepen.

## Regressie (Analyse van 2 kwantitatieve variabelen)

Dit betreft het onderzoeken van het verband tussen twee kwantitatieve variabelen.

### Regressie-analyse

Gebruik regressie-analyse wanneer je het verband tussen een afhankelijke variabele (y) en onafhankelijke variabelen (x) wilt begrijpen, en de waarde van de afhankelijke variabele wilt voorspellen. De afhankelijke variabele is 'y' en de onafhankelijke variabele is 'x'.

#### Correlatiecoëfficiënt (r)

Meet de sterkte en richting van de lineaire relatie tussen twee variabelen.

- Een r van 0 duidt op geen correlatie (punten liggen verspreid).
- Een r van 1 duidt op een sterke positieve correlatie (punten liggen op 1 lijn stijgend).
- Een r van -1 duidt op een sterke negatieve correlatie (punten liggen op 1 lijn dalend).
- De absolute waarde van R ($abs(R)$) geeft de sterkte aan. Interpretaties variëren van < 0.3 (very weak) tot > 0.95 (exceptionally strong). Een R > 0 duidt op een stijgend verband.

#### Coëfficiënt van determinatie (R-squared of r²)

Wordt gebruikt om de model fit te beoordelen, modellen te vergelijken, en het deel van de variantie te interpreteren dat verklaard wordt door de onafhankelijke variabelen.

- Een r² van 0 duidt op een zwakke correlatie/model fit.
- Een r² van 1 duidt op een sterke correlatie/model fit.
- De R² waarde geeft aan welk percentage van de variantie in de afhankelijke variabele verklaard wordt door de onafhankelijke variabele(n). Bijvoorbeeld, R² ≈ 0.64 betekent dat 64% van de variantie in Bwt wordt verklaard door Hwt.

## Tijdreeksanalyse

Tijdreeksanalyse is de analyse van data die geordend is in tijd over een bepaalde periode. Het doel is vaak om voorspellingen te maken voor de toekomst.

### Components of Time Series

Tijdreeksen kunnen de volgende componenten bevatten:

#### Trend

- De algemene richting van de data (stijgend, dalend, constant).

#### Seasonality

- herhaling van patronen in de data die op een vaste tijdsinterval gebeuren (bv. elke maandag een piek).

#### Noise

- onregelmatigheden in de data (random variaties).

#### Cyclical

- herhaling van patronen die niet op een vaste tijdsinterval gebeuren.

#### Moving Averages

- Methoden zoals simple moving average, weighted moving average, en exponential moving average. Weighted moving average geeft recentere data meer gewicht. Bij exponential moving average bepalen de $\alpha$ waarde en $1-\alpha$ hoe snel de gewichten afnemen en hoe groot de invloed van oudere observaties is op de voorspelling.

#### Exponential Smoothing

- Methoden die worden gekozen op basis van de aanwezigheid van trend en seasonality in de tijdreeks:

#### Single exponential smoothing (ook bekend als exponential moving average)

- Te gebruiken bij tijdreeksen met geen trend of seasonality.

#### Double exponential smoothing (Holt's method)

- Te gebruiken bij tijdreeksen met trend maar geen seasonality.

#### Triple exponential smoothing (Holt-Winters method)

- Te gebruiken bij tijdreeksen met trend en seasonality. Seasonality kan additief (constant) of multiplicatief (niet constant) zijn.
