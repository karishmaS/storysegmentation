#extract broadcast stream text
from lxml import etree
import sys
import os
datasetfolder = 'ldc_datasets/tdt2_careful_text/transcripts/'
datafilesnames = []

def extractfulltexts():
        extracttext('trainfiles', 'extracts')
        extracttext('testfiles', 'testextracts')
        
def extracttext(readfromfilename, savetofilename):
        with open(readfromfilename, 'r') as f:
                datafilesnames = f.readlines()
                f.close()
        with open(savetofilename, 'w') as w:
                for file_name in datafilesnames:
                        file_name = file_name.strip("\n")
                        filepath = os.path.abspath(os.path.join(os.pardir, datasetfolder+file_name))
                        with open(filepath, 'r') as f:
                                parsedPage = etree.HTML(f.read())
                                textdata = parsedPage.xpath("//turn//text()")
                                for line in textdata:
                                        line = line.replace('\n','')
                                        line = line.replace('\t','')
                                        w.write(line)
                                f.close()
                                w.write("\n---------------------------\n")
                w.close()
                print 'Extraction Complete:' + readfromfilename


extractfulltexts()
