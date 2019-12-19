
UTILISATION :

L'execution de l'un des trois algorithmes passe par le "main" et nécessite de fournir quatre paramètres :
- Un nom spécifique pour retrouver les résultats de l'instance lancée dans les dossiers de log
- Le nombre de générations
- Le nombre d'individu dans la population
- Le nom de l'algorithme choisi, parmi 'SHINE','MAPelites' et 'NS'

Par exemple, la ligne suivante lance SHINE sur 30 générations avec 150 individus :
$ python main.py TEST 30 150 SHINE

Le fichier de log sera créé dans le dossier "log/SHINE/" et son nom sera "SHINE_TEST_gen_30_size_150". Une image résultant de l'exploration sera également fournie.