import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,help='Folder that contains the EPIC.csv data')
parser.add_argument('--local',type=str,help='Folder that the python script is being run in')
args = parser.parse_args()
path=args.path
local=args.local
print('The path is: %s and the local is %s' % (path,local) )

#import sys
import re
from extract_dates_script import findDates
import os
import pandas as pd
cn_cols = ['Note.Data_ED.Notes','Note.Data_ED.Procedure.Note',
           'Note.Data_ED.Provider.Notes','Note.Data_ED.Triage.Notes']
df = pd.read_csv(path+'/EPIC.csv',encoding='ISO-8859-1',usecols=cn_cols)
ED_notes = df['Note.Data_ED.Notes']
procedure_notes = df['Note.Data_ED.Procedure.Note']
provider_notes = df['Note.Data_ED.Provider.Notes']
triage_notes = df['Note.Data_ED.Triage.Notes']

# Set to local
os.chdir(local)

# Make directory annotatorInput in the path folder if it does not exist
if not pd.Series(os.listdir(path)).isin(['annotatorInput']).any():
    print('---- Creating folder annotatorInput ----')
    os.mkdir(path+'/annotatorInput')

# Remove any files if there are any
for f in os.listdir(path+'/annotatorInput'):
    os.remove(path+'/annotatorInput'+'/' + f)

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
    text = text.replace("@", "at")
    
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

# Loop over each file and write to a csv
for ii in range(df.shape[0]):
    if ii % 10000 == 0:
        print('Iteration %i of %i' % (ii, df.shape[0]))
    # Clean text
    ED_note_ii = clean_text(ED_notes[ii])
    provider_notes_ii = clean_text(provider_notes[ii])
    procedure_notes_ii = clean_text(procedure_notes[ii])
    triage_notes_ii = clean_text(triage_notes[ii])
    # Write
    txt_writer(fn=path+'/annotatorInput/ED'+str(ii)+'.txt',ss=ED_note_ii)
    txt_writer(fn=path+'/annotatorInput/provider'+str(ii)+'.txt',ss=provider_notes_ii)
    txt_writer(fn=path+'/annotatorInput/procedure'+str(ii)+'.txt',ss=procedure_notes_ii)
    txt_writer(fn=path+'/annotatorInput/triage'+str(ii)+'.txt',ss=triage_notes_ii)
    

#dates = []
##clean notes in each column
#clean_ED = [clean(str(s)) for s in ED_notes]
#clean_procedure = [clean(str(s)) for s in procedure_notes]
#clean_provider = [clean(str(s)) for s in provider_notes]
#clean_triage = [clean(str(s)) for s in triage_notes]
#
##write clean text to files
#for i,s in enumerate(clean_ED):
#	file = open(r"annotatorInput/ED"+str(i)+".txt","w+")
#	file.write(s)
#	file.close()
#	
#for i,s in enumerate(clean_procedure):
#	file = open(r"annotatorInput/procedure"+str(i)+".txt","w+")
#	file.write(s)
#	file.close()
#
#for i,s in enumerate(clean_provider):
#	file = open(r"annotatorInput/provider"+str(i)+".txt","w+")
#	file.write(s)
#	file.close()
#
#for i,s in enumerate(clean_triage):	
#	file = open(r"annotatorInput/triage"+str(i)+".txt","w+")
#	file.write(s)
#	file.close()