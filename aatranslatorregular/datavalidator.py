# chr(number) returns the character that corresponds to the number. In c# it would be char c = (char)number
# ord(character) returns the number that corresponds to the character. In c# it would be int number = (int)character
import time
import gzip


class Validator:

    def __init__(self,path):
        self.path = path
        is_uneven = True
        line1 = ""
        line2 = ""
        self.char_dict : dict[str,int] = {}
        with gzip.open(self.path, "rt", encoding="utf-8-sig") as file:
            for line in file:
                if is_uneven:
                    line1 = line.strip()
                    is_uneven = False

                else:
                    line2 = line.strip()
                    self.convert_line(line1)
                    self.convert_line(line2)
                    is_uneven = True
        # print the dictionary
        for key in self.char_dict:
            print(f'{key} : {self.char_dict[key]}')
        # print total number of different characters
        print(f'Total number of different characters: {len(self.char_dict)}')
        # create single string from the dictionary
        char_string = ""
        for key in self.char_dict:
            char_string += key
        print(char_string)
        self.converter=Converter()


    def convert_line(self,line):
        for c in line:
            # if the character is not in the dictionary, add it
            if c not in self.char_dict:
                self.char_dict[c] = 1
            else:
                self.char_dict[c] += 1
    def test_converter(self):
        with gzip.open(self.path, "rt", encoding="utf-8-sig") as file:
            for line in file:
                # convert the line to a list of integers
                self.test_str(line.strip())
        print("Conversion successful")
    def test_str(self,line:str):
        intlist=self.converter.convert_2_list(line)
        # convert the list of integers back to a string
        str2=self.converter.convert_2_str(intlist)
        # compare the two strings
        if line!=str2:
            print("Conversion error")
            print(line)
            print(str2)
            raise Exception("Conversion error")


class Converter:
    def __init__(self):
        self.AA='_><ASGVIKCWEPLHRTDYFQNM'
        self.length=len(self.AA)
        self.AA2Index=[]
        self.Index2AA=[]
        high = 0
        for c in self.AA:
            if ord(c) > high:
                high = ord(c)
        self.AA2Index = [-1]*(high+1)
        cntr=0
        for c in self.AA:
            self.AA2Index[ord(c)]=cntr
            self.Index2AA.append(c)
            cntr+=1
        self.BOS=self.AA2Index[ord('>')]
        self.EOS=self.AA2Index[ord('<')]
        self.PAD=self.AA2Index[ord('_')]
    def convert_2_list(self,line:str,append_indicators:bool=False)->list[int]:
        int_list = []
        if append_indicators:
            int_list.append(self.BOS)
        for c in line:
            int_list.append(self.AA2Index[ord(c)])
        if append_indicators:
            int_list.append(self.EOS)
        return int_list
    def convert_2_str(self,int_list:list[int])->str:
        line = ""
        for i in int_list:
            line += self.Index2AA[i]
        return line
    









