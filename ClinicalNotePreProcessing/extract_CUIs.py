#opening csv file to determine lengths of the four columns
import pandas
df = pandas.read_csv('../ED_data/EPIC.csv')
ED_notes = [None] * len(df['Note.Data_ED.Notes'])
procedure_notes = [None] * len(df['Note.Data_ED.Procedure.Note'])
provider_notes = [None] * len(df['Note.Data_ED.Provider.Notes'])
triage_notes = [None] * len(df['Note.Data_ED.Triage.Notes'])

labels = ["ED", "procedure", "provider", "triage"]

import re

def read_output_files():
	
	#loop through each column
	for x,col in enumerate([ED_notes, procedure_notes, provider_notes, triage_notes]):
		for i,s in enumerate(col):
			cui_list = []
			
			#open and read file with annotations
			fileName = r"annotatorOutput/"+labels[x]+str(i)+".txt.xmi"
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
			col[i] = cui_list
	
	return
	
read_output_files()

#add four columns with CUI codes to the csv file
df['Processed.Note.Data_ED.Notes'] = ED_notes
df['Processed.Note.Data_ED.Procedure.Note'] = procedure_notes
df['Processed.Note.Data_ED.Provider.Notes'] = provider_notes
df['Processed.Note.Data_ED.Triage.Notes'] = triage_notes

df.to_csv('../ED_data/Processed_EPIC.csv', index=False)



