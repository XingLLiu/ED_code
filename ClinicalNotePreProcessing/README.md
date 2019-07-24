<b>Running Pre-Processing Pipeline:</b>
<br><br>
<b>Getting a UMLS license:</b>
<br> - apply for a license at <br>https://uts.nlm.nih.gov/license.html
<br><br>
<b>Installing cTAKES 4.0:</b>
<br>  - ensure that you have a JDK of version 1.8 or higher
<br>  - download from <br>http://ctakes.apache.org/downloads.html
<br>  - refer to User Installation Guide at <br>https://cwiki.apache.org/confluence/display/CTAKES/cTAKES+4.0+User+Install+Guide#cTAKES4.0UserInstallGuide-InstallcTAKES<br> for installation instructions
<br>  - download cTAKES resources from <br>https://sourceforge.net/projects/ctakesresources/files/
<br>  - refer back to User Installation Guide at <br>https://cwiki.apache.org/confluence/display/CTAKES/cTAKES+4.0+User+Install+Guide#cTAKES4.0UserInstallGuide-InstallcTAKES<br> for resource installation instructions
<br><br>
<b>Getting Ready to run the Pipeline:</b>
<br>- modify clean_extract_notes.sh to have your UMLS license username and password, the absolute path locations storing the annotatorInput and annotatorOutput folders (should be in ED_code), and the absolute path location where your cTAKES installation is located
<br>- check that cleaning_script.py and extract_CUIs.py read and write to the correct csv file
