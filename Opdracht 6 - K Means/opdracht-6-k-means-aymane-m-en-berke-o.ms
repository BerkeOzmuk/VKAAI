.TL
Opdracht 6 - K-Means
.AU
Aymane Machrouki en Berke Ozmuk
.NH 1
Call tree
.PP
main()
.PP
-----importData()
.PP
-----createLabels()
.PP
-----normalizeData()
.PP
-----kMeans()
.PP
----------centroidAvg()
.PP
----------getCentroids()
.PP
-----assignLabels()
.PP
-----countLabels()
.PP
-----createSkreePlot()
.PP
.NH 1
Totaal-uitleg
.RS
Als eerste wordt de data uit dataset1.csv geimporteerd door
.B importData()
, waarna het direct wordt genormaliseerd met 
.B normalizeData()
.
Hierna wordt de 
.B kMeans() 
functie aangeroepen met de gekozen k.
In de functie worden random centroids gekozen waarvan het aantal besloten wordt door de meegegeven k door de
.B getCentroids() 
functie aan te roepen.
Na de centroids te hebben gekozen wordt er door alle datapunten gelooped en wordt er gechecked welke centroid het dichtste bij is.
Als alle datapunten zijn gechecked, dan worden ze in clusters opgedeeld.
Aan het eind van de functie wordt het gemiddelde van elk cluster berekend met behulp van de 
.B centroidAvg()
functie, waarna deze gemiddelden gebruikt worden als de nieuwe centroids.
Na dit te hebben gedaan start de funtie opnieuw op, met de nieuwe centroids in de plaats van de oude.
De functie blijft dit herhalen totdat er geen vernadering meer is.
Nadat de functie is afgerond wordt de 
.B assignLabels()
functie aangeroepen met de resulaten van de 
.B kMeans()
functie.
De 
.B assignLabels()
functie voegt het bijbehorende label toe aan elk datapunt.
Hierna wordt er geteld welke seizoenen er in elk cluster voorkomen met de 
.B countLabels() 
functie.
Als laatste wordt de 
.B createSkreePlot() 
functie aangeroepen om een screeplot te maken van de meegeleverde k.
.RE
.NH 1
Resulaten
.PP
Gegeven dat k 4 is, geven de resultaten aan dat...
.PP
	...cluster 1 53 winters, 21 lentes, 0 zomers en 3 herfsten bevat
.PP
	...cluster 2 0 winters, 14 lentes, 38 zomers en 8 herfsten bevat
.PP
	...cluster 3 30 winters, 36 lentes, 7 zomers en 39 herfsten bevat
.PP
	...cluster 4 8 winters, 21 lentes, 47 zomers en 41 herfsten bevat
.PP
.PP
Op de volgende paginas zijn 2 skree plots te zien.
.PDFPIC "scree-plot-1.pdf"
Bij deze skree plot is er een k van 5 gekozen.
.PDFPIC "scree-plot-2.pdf"
Bij deze skree plot is er een k van 10 gekozen.
.NH 1
Antwoorden
.PP
Hoeveel clusters kan je (betrouwelijk) detecteren?

.RS
Door de twee skree plots met elkaar te vergelijken is te zien dat na een k van 6 en 7 de plot al begint af te vlakken.
Daarom zou een k van 6 optimaal moeten zijn.
.RE
