(Stand: 17.03.20, aktuell wird ein neues news System mit Flask und Datenbanken erstellt) Aktuell gibt es unter *reed:/var/www/www\_scripts/* zwei Arten von news items.

---++ Aktuelles

1. Kleine Snippets für die rechte Seitenleiste auf [https://www.cl.uni-heidelberg.de](unserer Homepage), hier werden die meisten angefragten Neuigkeiten reingestellt, also standardmäßig Sekretariats Announcements von Sandra oder andere Neuigkeiten, Publikationen und Data Releases. Die Links kommen außerdem noch auf [https://www.cl.uni-heidelberg.de/studies/](die Seite "Im Studium").
Der Unterordner hierfür ist *aktuelles/*.

   Für eine neue Notiz ist hier der Ablauf wie folgt:
      
      i. in den Unterordner *items/* gehen.
      i. Die Zahlen XY in den Dateinamen heißen: 
      i. X: Kategorie; dabei steht 0 für templates,
         1.  für Aktuelles (mit RSS Feed),
         2.  für Data Releases
         3.  für Publikationen
      i. Y: absteigende Auflistung auf der Seitenleiste: 0 zuoberst, 9 zuletzt
      i. neue Datei im Stil der dort vorhandenen anlegen (einfach nur einen *a* link tag). Verweisen auf eine neue Datei unter */var/www/htdocs/news/* für Aktuelles und Data Releases bzw auf eine neue unter */var/www/htdocs/news/publications/* für Publikationen. Entsprechend die Datei benennen.
      i. alte Dateien und Nummerierung entsprechend anpassen 
      i. Jetzt die Datei erstellen in */var/www/htdocs/news/*(*publications/*), dafür im jeweiligen Ordner z.B. *template_pubs.mhtml* ausfüllen.
      i. Auf [https://www.cl.uni-heidelberg.de/news/template_news.mhtml](dem entsprechenden Link) überprüfen, ob alles stimmt.
      i. zurück in *var/www/www_scripts/aktuelles/* gehen und den command *./runScript* ausführen.
      i. Jetzt sollte auf [https://www.cl.uni-heidelberg.de/](unserer Homepage) alles stehen. 
      i. ggf Kalendereintrag zum Entfernen machen :-)  
      
   Für das Entfernen eines alten Eintrags ganz einfach diese beiden Commands ausführen:
   
      i. *mv items/13.old_notice.html items/Archiv/*
      i. *./runScript*
      i. Zahlen anderer Einträge müssen nicht angepasst werden da nur größer/kleiner gecheckt wird.

