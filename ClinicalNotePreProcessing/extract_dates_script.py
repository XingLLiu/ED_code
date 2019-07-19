import datetime
import re

months_abbrv = {'jan': '01', 'feb':'02', 'mar':'03', 'apr':'04', 'may':'05', 'jun':'06', 'jul':'07', 'aug':'08', 'sept':'09', 'oct':'10', 'nov':'11', 'dec':'12'}
months_full = {'january': '01', 'february':'02', 'march':'03', 'april':'04', 'may':'05', 'june':'06', 'july':'07', 'august':'08', 'september':'09', 'october':'10', 'november':'11', 'december':'12'}

#this function extracts the dates located in text and returns a list with the list of dates at its first index and the text with all of its dates rewritten in the same format at its second index
def findDates (text):
    #date formats:
         #dd/mm/yy - pattern 1 
         #dd/mm/yyyy - pattern 2 
         #mon date yy - pattern 3 
         #mon date year - pattern 4 
         #mon date(abbrev) yy - pattern 5 
         #mon date(abbrev) year - pattern 6 
         #mon date - pattern 7 
         #mon date(abbrev) - pattern 8 
         #month date yy - pattern 9 
         #month date year - pattern 10 
         #month date(abbrev) yy - pattern 11 
         #month date(abbrev) year - pattern 12 
         #month date - pattern 13 
         #month date(abbrev) - pattern 14 
    
    dates_found = []
    
    pattern_1 = re.findall ("((?<!\d)\d\d|^\d\d)/(\d\d)/(\d\d(?!\d)|\d\d$)", text)
    for p in pattern_1:
        date_found = (datetime.date(int('20' + p[2]), int(p[1]), int(p[0])))
        dates_found.append (date_found)
         

    pattern_2 = re.findall ("((?<!\d)\d\d|^\d\d)/(\d\d)/(\d{4,4}(?!\d)|\d\d\d\d$)", text)
    for p in pattern_2:
        text = text.replace (p[0]+ "/" + p[1] + "/" + p[2], p[0]+ "/" + p[1] + "/" + (p[2][-2:]))         

    pattern_3 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,3}) *([\d]{1,2}) +(\d{2,2}(?!\d)|\d\d$)", text)
    for p in pattern_3:
        if (p[0]).lower() in months_abbrv:
            text = re.sub (p[0] + " *" + p[1]+ " +" + p[2], p[1] + "/" + months_abbrv[(p[0]).lower()] + "/" + p[2], text)

    pattern_4 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,3}) *([\d]{1,2}) +(\d{4,4})[^\d]", text)
    for p in pattern_4:
        if (p[0]).lower() in months_abbrv:
            text = re.sub (p[0] + " *" + p[1] + " +" + p[2], p[1] + "/" + months_abbrv[(p[0]).lower()] + "/" + p[2][-2:], text)

    pattern_5 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,3}) *([\d]{1,2}) *[a-zA-Z]{2,2} *(\d{2,2}(?!\d)|\d\d$)", text)
    for p in pattern_5:
        if (p[0]).lower() in months_abbrv:
            text = re.sub (p[0] + " *" + p[1] + " *[a-zA-Z]{2,2} *" + p[2], p[1] + "/" + months_abbrv[(p[0]).lower()] + "/" + p[2], text)

    pattern_6 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,3}) *([\d]{1,2}) *[a-zA-Z]{2,2} *(\d{4,4}(?!\d)|\d\d\d\d$)", text)
    for p in pattern_6:
        if (p[0]).lower() in months_abbrv:
            text = re.sub (p[0] + " *" + p[1] + " *[a-zA-Z]{2,2} *" + p[2], p[1] + "/" + months_abbrv[(p[0]).lower()] + "/" + p[2][-2:], text)


    pattern_10 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,9}) *([\d]{1,2}) +(\d{4,4})[^\d]", text)
    for p in pattern_10:
        if (p[0]).lower() in months_full:
            text = re.sub (p[0] + " *" + p[1] + " +" + p[2], p[1] + "/" + months_full[(p[0]).lower()] + "/" + p[2] [-2:], text)
             
    pattern_9 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,9}) *([\d]{1,2}) +(\d\d)[^\d]", text)
    for p in pattern_9:
        if (p[0]).lower() in months_full:
            text= re.sub (p[0] + " *" + p[1] + " +" + p[2], p[1] + "/" + months_full[(p[0]).lower()] + "/" + p[2], text)

   
    pattern_11 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,9}) *([\d]{1,2}) *[a-zA-Z]{2,2} *(\d{2,2}(?!\d)|\d\d$)", text)
    for p in pattern_11:
        if (p[0]).lower() in months_full:
            text = re.sub (p[0] + " *" + p[1] + " *[a-zA-Z]{2,2} *" + p[2], p[1] + "/" + months_full[(p[0]).lower()] + "/" + p[2], text)
             

    pattern_12 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,9}) *([\d]{1,2}) *[a-zA-Z]{2,2} *(\d{4,4}(?!\d)|\d\d\d\d$)", text)
    for p in pattern_12:
        if (p[0]).lower() in months_full:
            text = re.sub (p[0] + " *" + p[1] + " *[a-zA-Z]{2,2} *"+ p[2], p[1] + "/" + months_full[(p[0]).lower()] + "/" + (p[2])[-2:], text)
             
    pattern_14 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,9}) *([\d]{1,2}) *[a-zA-Z]{2,2} *(?!\d)", text)
    for p in pattern_14:
        if (p[0]).lower() in months_full:
            if len(dates_found) > 0:
                yr = (dates_found[0]).year
            else:
                yr = datetime.datetime.now().year
                
            text = re.sub (p[0] + " *" + p[1] + " *[a-zA-Z]{2,2} *(?!\d)", p[1]+"/"+months_full[(p[0]).lower()]+ "/"+(str(yr)[-2:]), text)

    pattern_13 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,9}) *([\d]{1,2}) *(?!\d)", text)
    for p in pattern_13:
        if (p[0]).lower() in months_full:
            if len(dates_found) > 0:
                yr = (dates_found[0]).year
            else:
                yr = datetime.datetime.now().year
            text = re.sub (p[0] + " *" + p[1]+" *(?!\d)", p[1] + "/" + months_full[(p[0]).lower()] + "/" + (str(yr))[-2:], text)
      
			
    pattern_8 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,3}) *([\d]{1,2}) *[a-zA-Z]{2,2}(?!\d)|$", text)
    for p in pattern_8:
        if (p[0]).lower() in months_abbrv:
            if len(dates_found) > 0:
                yr = (dates_found[0]).year
            else:
                yr = datetime.datetime.now().year
            text = re.sub (p[0] + " *" + p[1] + " *[a-zA-Z]{2,2}(?!\d)|$", p[1] + "/" + months_abbrv[(p[0]).lower()] + "/" + (str(yr))[-2:], text)

			
    pattern_7 = re.findall ("[^a-zA-Z0-9]([a-zA-Z]{3,3}) *([\d]{1,2}) *(?!\d)", text)
    for p in pattern_7:
        if (p[0]).lower() in months_abbrv:
            if len(dates_found) > 0:
                yr = (dates_found[0]).year
            else:
                yr = datetime.datetime.now().year
            text = re.sub (p[0] + " *" + p[1], p[1] + "/" + months_abbrv[(p[0]).lower()] + "/" + (str(yr))[-2:], text)


    #collect and remove all dates
    dates_found = []
    allDates = re.findall ("((?<!\d)\d{1,2}|^\d{1,2})/(\d{1,2})/(\d{1,2}(?!\d)|\d{1,2}$)", text)
    for a in allDates:
        date_found = (datetime.date(int('20' + a[2]), int(a[1]), int(a[0])))
        dates_found.append (date_found)
        if date_found.day < 10:
            newDate = '0'+str(date_found.day)
        else: newDate = (date_found.day)
        if date_found.month < 10:
            newMonth = '0'+str(date_found.month)
        else: newMonth = (date_found.month)
                
        text = re.sub (a[0] + "/" + a[1] + "/" + a[2], str(newDate)+"/"+str(newMonth)+"/"+str(date_found.year)[-2:], text)    
            
    return [dates_found, text]
