# Varsinainen projektihakemisto alkaa tästä.
## Sisältö:
* `/Facerec`
  * `firstversion.py` -- Ensinmäinen versio/luonnos varsinaisesta kasvojentunnistusapplikaatiosta.
  * `heimdall.py` -- Viimeinen, lopullinen versio kasvojentunnistusapplikaatiosta.
  * `trainer.py` -- Python ohjelma joka "opetti". Datasetistä -> `trained.yml`
* `/dataset`
  * `otakuvia.py`  Ottaa kuvia n-määrän n-tiheydellä. Skaalaa kuvat sopiviksi ja muuttaa värimaailman Haar Cascadelle sopivaksi.
  * `/kuvat/final`-hakemisto sisältää viimeisimmän version kuvat jotka opetetaan `trainer.py`:lla.
* `/Cascades`
  * Sisältää koneoppimisen objektintunnistusalgoritmin, valinta ja tyyli on vapaa. Käytimme [Haar Cascadea.](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
* `/Training`
  * Sisältää opetetun materiaalin mitä hyödynnetään `heimdall.py`:ssä.
