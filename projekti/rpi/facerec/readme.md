# Varsinainen projektihakemisto alkaa tästä.
## Sisältö:
* `/facerec`
  * `firstversion.py` -- Ensinmäinen versio/luonnos varsinaisesta kasvojentunnistusapplikaatiosta.
  * `heimdall.py` -- Viimeinen, lopullinen versio kasvojentunnistusapplikaatiosta.
  * `trainer.py` -- Python ohjelma joka "opetti". Datasetistä -> `trained.yml`
* `/facerec/dataset`
  * `otakuvia.py` -- Ottaa kuvia n-määrän n-tiheydellä. Skaalaa kuvat sopiviksi ja muuttaa värimaailman Haar Cascadelle sopivaksi.
  * `/facerec/dataset/kuvat/final` -- Hakemisto sisältää viimeisimmän version kuvat jotka opetetaan `trainer.py`:lla.
  * Ei sisällä kuvia täällä githubissa, sehän ei hyödyttäisi ketään.
* `/facerec/Cascades`
  * Sisältää koneoppimisen objektintunnistusalgoritmin, valinta ja tyyli on vapaa. Käytimme [Haar Cascadea.](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
* `/Training`
  * Sisältää opetetun materiaalin `trained.yml` mitä hyödynnetään `heimdall.py`:ssä.
  * valmista opetusdataa täällä ei ole.
