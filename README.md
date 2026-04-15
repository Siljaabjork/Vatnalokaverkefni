# Vatnalokaverkefni

## Lýsing

Þetta verkefni inniheldur vatnafræðilega greiningu á rennsli Dynjandisár sem er vatnasvið ID_12 úr LamaH-Ice gagnasafninu.
Markmiðið er að greina rennslishegðun, flóðatíðni, leitni o.fl. og búa til myndir sem eru notaðar í skýrslu.

## Gögn

Notuð eru gögn úr **LamaH-Ice gagnasafninu**, meðal annars:

* Daglegt rennsli (`qobs`)
* Úrkoma (`prec`)
* Hitastig (`2m_temp_mean`)
* Lýsigögn vatnasviðs

Gögnin eru **ekki geymd í repo**.
Hægt er að nálgast þau hér:
https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/

## Mappaskipan

```
Vatnalokaverkefni/
│
├── data/           # Unnin gögn
├── lamah_ice/      # Gagnasafn (ekki í repo)
├── scripts/        # Python kóði
│   ├── lidur2.py
│   ├── lidur3.py
│   ├── lidur4.py
│   ├── lidur5.py
│   ├── lidur6.py
│   ├── lidur7.py
│   ├── lidur8.py
│   └── main.py     # Keyrsluskrá
├── figures/        # Myndir sem kóðinn býr til
└── README.md
```

## Keyrsla

Verkefnið er keyrt með:

```bash
python scripts/main.py
```

Í `main.py` er hægt að velja hvaða liði á að keyra með því að uncommenta viðeigandi línu:

```python
run_lidur2()
run_lidur3()
run_lidur4()
run_lidur5()
run_lidur6()
run_lidur7()
run_lidur8()
```

## Niðurstöður / myndir

Kóðinn býr til myndir í möppunni `figures/`, t.d.:

* meðaltalsár rennslis, úrkomu og hitastigs
* flow duration curve
* flóðatíðnigreining
* leitni (árleg og árstíðabundin)
* rennslisatburði

## Kröfur

Python pakkar sem þarf að hafa:

* pandas
* numpy
* matplotlib
* scipy
* pymannkendall

## Athugasemdir

* Gögn úr LamaH-Ice eru ekki innifalin vegna stærðar þeirra
* Kóðinn gerir ráð fyrir að gagnamappan `lamah_ice/` sé til staðar með réttum slóðum
* Allar niðurstöður eru endurgeranlegar með því að keyra `main.py`

## Höfundar

Hólmfríður Hermannsdóttir,
Sigríður Lára Ingibjörnsdóttir og
Silja Björk Ægisdóttir
