# ----------------------------------------------------
# Input format of clinical notes:
# Three strings:
# 1. dd/mm/yyyy tttt, Some Name
# 2. description of patient
# 3. description of symptoms
# ----------------------------------------------------
import os
import sys
import subprocess
import re
import pandas as pd

# Change to the previous folder to import pre-written modules
sys.path.append("../")
from extract_dates_script import findDates

# Make directory annotatorInput in the path folder if it does not exist
cwd = os.getcwd()
if not os.path.exists(cwd):
    print('---- Creating folder annotatorInput ----')
    os.makedirs(cwd + '/annotatorInput')


# ----------------------------------------------------

# Raw input of clinical note
raw_note = sys.argv
# Change note to input format
prefix1 = "<<STARTNOTE "
prefix2 = ", RN<NOTETEXT "
prefix3 = " <CRLF>"
suffix = "<CRLF> NOTETEXT>ENDNOTE>>"

if len(raw_note[1]) == 0:
    raw_note[1] = "01/04/1875 0801,Emergency,John Smith"


if len(raw_note[2]) == 0:
    raw_note[2] = "In the morning in the sun all, fell in the kiddie pool. Went to Nowhere Hospital yesterday."


if len(raw_note[3]) == 0:
    raw_note[3] = "Fever Tmax 39.2 C started today 3-4 hours after fell in pool"


note = prefix1 + raw_note[1] + prefix2 + raw_note[2] + prefix3 + raw_note[3] + suffix 


# ----------------------------------------------------
# Cleaning script
#a list of common abbreviations that do not have other potential meanings
abbrevs = {'hrs':'hours', 'mins':'minutes',
           'S&S':'signs and symptoms', 
           'bc':'because', 'b/c':'because', 
           'wo':'without', 'w/o':'without', 
           'yo':'year old', 'y.o':'year old', 'wk':'weeks',
           'm.o':'month old', 'mo':'months', 'mos':'months', 
           'b4':'before', 'pt':'patient',
           'ro':'rule out', 'w/':'with', 
           'o/n':'overnight', 'f/u':'follow up',
           'M':'male', 'F':'female'}

# Function that cleans the text
def clean_text(text):
    if ((text == 'nan') | (text != text)):
        return ''
    #date extraction and replacement
    # dates = findDates(text)[0] # USE ME PLEASE!
    text = findDates(text)[1]

    #note structure
    text = text.replace ("," , " ")
    text = re.sub (" *<<STARTNOTE.*<NOTETEXT", "", text)
    text = text.replace("NOTETEXT>ENDNOTE>>", " ")
    text = re.sub (" *<CRLF>", ". ", text)
    #erroneous UTF symbols
    text = re.sub ("[•â€¢Ã]+", "", text)
    #abbreviations
    for abb in abbrevs:
        if " " + abb + "." in text:
            text = text.replace (" " + abb + ".", " " + abbrevs[abb] + ".")
        elif  " " + abb + " " in text:
            text = text.replace (" " + abb + " ", " " + abbrevs[abb] + " ")
    #numeric ranges
    grp = re.findall ("(?<![0-9]-)([0-9]+) *- *([0-9]+)(?!-[0-9])", text)
    for g in grp:
        text = re.sub ("(?<![0-9]-)" + g[0]+" *- *" + g[1] + "(?!-[0-9])", g[0] + " to " + g[1], text)
    #dealing with x[0-9]+ d
    grp = re.findall("x *([0-9]+) *d([ .,]+)", text)
    for g in grp:
        text = re.sub ("x *" + g[0] + " *d"+g[1], "for " + g[0] + " days" + g[1], text)
    grp = re.findall("x *([0-9]+)/d([ .,]+)", text)
    for g in grp:
        text = re.sub ("x *" + g[0] + "/d"+g[1], g[0] + " times per day" + g[1], text)
    grp = re.findall("x *([0-9]+)/day([ .,]+)", text)
    for g in grp:
        text = re.sub ("x *" + g[0] + "/day" + g[1], g[0] + " times per day" + g[1], text)       
    #dealing with multiple plus signs
    grp = re.findall ("([a-zA-Z0-9]*) *\+{2,3}", text)
    for g in grp:
        text = re.sub (g + " *\+{2,3}", "significant " + g, text)
    #switching symbols for equivalent words
    text = text.replace ("%", " percent ")
    text = text.replace ("=" , " is ")
    text = text.replace ("\$", " dollars ")
    text = text.replace (">", " greater than ")
    text = text.replace ("<", " less than ")
    text = text.replace ("?", " possible ")
    text = text.replace ("~", " approximately ")
    text = text.replace ("(!)", " abnormal ")
    #numeric ratios
    grp = re.findall ("(\d{1,1}) *\: *(\d{1,2}) *[^ap][^m][^a-zA-Z0-9]", text)
    for g in grp:
        text = re.sub (g[0] + " *: *" + g[1], g[0] + " to " + g[1] + " ratio", text)
	#symbol removal
    text = text.replace ("["," ")
    text = text.replace ("]"," ")
    text = text.replace ("{"," ")
    text = text.replace ("}"," ")
    text = text.replace ("\\"," ")
    text = text.replace ("|"," ")
    text = text.replace ("-"," ")
    text = text.replace ("_"," ")
    #extra spaces
    text = re.sub (" +", " ", text)
	#extra periods
    text = re.sub ("\. *\.[ .]+", ". ", text)
    return text


# Wrapper to write a text file
def txt_writer(fn,ss):
    file = open(fn,'w+')
    file.write(ss)
    file.close()


# Clean text
cleaned_note = clean_text(note)
print('Cleaned note:\n', cleaned_note)


# Output cleaned note
txt_writer(fn=cwd+'/annotatorInput/ED'+'.txt',ss=cleaned_note)

# Change working directory to Apache
home_dir = cwd.split('/')[0:3]
# os.chdir('/'.join(home_dir) + '/Documents/Ctakes/apache-ctakes-4.0.0/bin')
subprocess.call([cwd + '/input_example.sh'])



# ----------------------------------------------------
# Extract CUIs
cui_list = []
#read file with annotations
fileName = cwd+"/annotatorOutput/ED.txt.xmi"
file = open(fileName,"r")
text = file.read()
#find all annotations with UMLS references
concept_nums = re.findall ("ontologyConceptArr=\"(.*?)\"", text) 
polarity = re.findall ("ontologyConceptArr.*?polarity=\"(.+?)\"", text) #find negation value for those annotations		
#finding CUI codes for annotations
for y, n in enumerate(concept_nums):
    codes = n.split()
    c = codes[0] #eliminating repeated CUI codes for the same annotation
    if re.search("xmi\:id\=\"" + c + "\".*?cui=\"(C[0-9]{7,7})\"", text) != None:
        if polarity[y] == "1": #a polarity of 1 indicates no negation
            negation = ""
        else: #a polarity of -1 indicates a negation
            negation = "N"
        cui_list.append(negation + (re.findall("xmi:id=\"" + c + "\".*?cui=\"(C[0-9]{7,7})\"", text)[0]))
file.close()
file = ''
text = ''


#print out processed note and CUIs
print("----------------------------------------------------")
print("Clinical note: \n{} \n\nProcessed CUIs: \n{}".format("\n".join(raw_note[1:]), cui_list))

