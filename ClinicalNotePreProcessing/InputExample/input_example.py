'''
<<STARTNOTE 01/07/18 0022,Emergency,Alvarado, Marjorie, 
RN<NOTETEXT In the morning in the sun all, fell in the kiddie pool, face went into water, fell next to the slide, unsure if hit slide, c/o left ear now, had a slight cough for a week, feels like it has gotten worse, irritable, fussy, went to hospital- Markham Stouffville Hospital today for ear infection, did not mention at that visit about the pool incident. Could not see if has ear infection because could not take wax out. 
<CRLF>Fever Tmax 39.6 C started today 2-3 hours after fell in pool<CRLF> NOTETEXT>ENDNOTE>>
'''

# ----------------------------------------------------
import sys
import subprocess
note = sys.argv[1]

# ----------------------------------------------------
# Cleaning script
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--path',type=str,help='Folder that contains the EPIC.csv data')
# parser.add_argument('--local',type=str,help='Folder that the python script is being run in')
# args = parser.parse_args()
# path=args.path
# local=args.local
# print('The path is: %s and the local is %s' % (path,local) )

import re
from extract_dates_script import findDates
import os
import pandas as pd
# cn_cols = ['Note.Data_ED.Notes','Note.Data_ED.Procedure.Note',
#            'Note.Data_ED.Provider.Notes','Note.Data_ED.Triage.Notes']
# df = pd.read_csv(path+'/EPIC.csv',encoding='ISO-8859-1',usecols=cn_cols)
# ED_notes = df['Note.Data_ED.Notes']
# procedure_notes = df['Note.Data_ED.Procedure.Note']
# provider_notes = df['Note.Data_ED.Provider.Notes']
# triage_notes = df['Note.Data_ED.Triage.Notes']

# # Set to local
# os.chdir(local)

# # Make directory annotatorInput in the path folder if it does not exist
# if not pd.Series(os.listdir(path)).isin(['annotatorInput']).any():
#     print('---- Creating folder annotatorInput ----')
#     os.mkdir(path+'/annotatorInput')

# # Remove any files if there are any
# for f in os.listdir(path+'/annotatorInput'):
#     os.remove(path+'/annotatorInput'+'/' + f)

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

# Loop over each file and write to a csv
# Clean text
cleaned_note = clean_text(note)
print('Cleaned note:\n', cleaned_note)
    


# Change working directory to Apache
cwd = os.getcwd().split('/')
txt_writer(fn=cwd+'/InputExample/annotatorInput/ED'+'.txt',ss=cleaned_note)
os.chdir('/'.join(cwd[1:3]) + '/Documents/Ctakes/apache-ctakes-4.0.0/bin')

subprocess.call(['./input_example.sh'])



# ----------------------------------------------------
# Extract CUIs
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--path',type=str,help='Folder that contains the EPIC.csv data')
# args = parser.parse_args()
# path=args.path
# print('The path is: %s' % (path) )

#opening csv file to determine lengths of the four columns
# import pandas as pd
# df = pd.read_csv(path+'/EPIC.csv',encoding='ISO-8859-1')
# ED_notes = [None] * len(df['Note.Data_ED.Notes'])
# procedure_notes = [None] * len(df['Note.Data_ED.Procedure.Note'])
# provider_notes = [None] * len(df['Note.Data_ED.Provider.Notes'])
# triage_notes = [None] * len(df['Note.Data_ED.Triage.Notes'])

# labels = ["ED", "procedure", "provider", "triage"]

import re

#loop through each column
for i,s in enumerate(col):
    cui_list = []
    #read file with annotations
    fileName = cwd+"InputExample"+"/annotatorOutput/"+labels[x]+str(i)+".txt.xmi"
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
            cui_list.append (negation + (re.findall("xmi:id=\"" + c + "\".*?cui=\"(C[0-9]{7,7})\"", text)[0]))
    file.close()
    file = ''
    text = ''
    # col[i] = cui_list

#def read_output_files(): return #read_output_files()

#add four columns with CUI codes to the csv file
df_CUI = pd.DataFrame({'CSN':df.CSN, 'MRN':df.MRN,
                      'Processed.Note.Data_ED.Notes':ED_notes,
                      'Processed.Note.Data_ED.Procedure.Note':procedure_notes,
                      'Processed.Note.Data_ED.Provider.Notes':provider_notes,
                      'Processed.Note.Data_ED.Triage.Notes':triage_notes})

df_CUI.to_csv(path+'/EPIC_CUIs.csv', index=False)
    
#df['Processed.Note.Data_ED.Notes'] = ED_notes
#df['Processed.Note.Data_ED.Procedure.Note'] = procedure_notes
#df['Processed.Note.Data_ED.Provider.Notes'] = provider_notes
#df['Processed.Note.Data_ED.Triage.Notes'] = triage_notes
#
#df.to_csv('../../Processed_EPIC.csv', index=False)