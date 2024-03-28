@echo off

set type=%1

set topic_path=%2

set wiki_dir=.\data

set jar_dir=.\topmost\evaluations

java -jar "%jar_dir%\pametto.jar" "%wiki_dir%\wikipedia\wikipedia_bd" %type% %topic_path%

