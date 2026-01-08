# Analyse: Vegvesen-tellestasjoner vs Ferde-bompasseringer (Ryfast)

Kilde/fil:
- Ferde: `docs/2025_2024 trafikk og inntekt_Ryfast (1).xlsx`
- Vegvesen (tellepunkter): `trafikkdata-api.atlas.vegvesen.no`
- Sammenstilling: `docs/vegvesen_vs_ferde_ryfast_2024_2025.csv`

## Hva som sammenlignes
- **Ferde**: bompasseringer (inkl. fritakspasseringer) og beregnet inntekt.
- **Vegvesen**: trafikktellinger i målepunkter (månedsvis gj.snitt per døgn × dager i måneden), summert over 6 punkter:
  - Ryfylketunnelen: `99040V2725982`, `00911V2725983`
  - Hundvågtunnelen (inkl. påramper): `10239V2725979`, `92743V2726085`, `62464V2725991`, `25926V2725990`

## Årsnivå
- **2024 (12/12 mnd med full punktdekning)**: Vegvesen `7 162 124` vs Ferde `6 940 889` → **+221 235** (+`3,19%`)
- **2025**:
  - **Observerte summer (12 rader)**: Vegvesen `7 233 497` vs Ferde `7 197 420` → **+36 077** (+`0,50%`)
  - **Sammenlignbare måneder (11/12 mnd, alle 6 punkter har data)**: Vegvesen `6 763 754` vs Ferde `6 520 524` → **+243 230** (+`3,73%`)
  - Årsavviket ser derfor “for lite” ut i 2025 fordi Vegvesen mangler data for august på 2 av målepunktene.

## Månedlige avvik (2024)
- Avviket er stabilt positivt (Vegvesen > Ferde) og følger månedsmønsteret tett (korrelasjon ~`0,998`).
- Største positive avvik i 2024 (Vegvesen − Ferde): september `+28 787`, mai `+25 701`, juli `+22 895`.

## Datakvalitet / manglende data (2025)
- **August 2025**: Vegvesen har data fra **4/6** målepunkter (kolonne `vegvesen_antall_punkt_med_data`), noe som gir et kunstig lavt Vegvesen-tall den måneden.
- Lavere gjennomsnittlig dekning (coverage) i Vegvesen-APIet finnes også i enkelte måneder i 2025 (se kolonne `vegvesen_dekning_snitt_prosent`).

## Tolkning (praktisk)
- Når vi sammenligner “like med like” (måneder der alle målepunktene har data), ligger Vegvesen systematisk rundt **3–4% høyere** enn Ferde i både 2024 og 2025.
- Dette kan skyldes at datasettene måler ulike ting (trafikktelling i punkt vs bompassering/transaksjon), eller at geometri/avgrensing ikke er helt identisk.
