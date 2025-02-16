import time
import gzip


class AADataSet:

    def __init__(self,path):
        self.path = path
        is_uneven = True
        line1 = ""
        line2 = ""
        self.char_dict : dict[str,int] = {}
        with gzip.open(self.path, "rt", encoding="utf-8") as file:
            for line in file:
                if is_uneven:
                    line1 = line
                    is_uneven = False

                else:
                    line2 = line
                    self.convert_line(line1)
                    self.convert_line(line2)
                    is_uneven = True
        # print the dictionary
        for key in self.char_dict:
            print(f'{key} : {self.char_dict[key]}')


    def convert_line(self,line):
        for c in line:
            # if the character is not in the dictionary, add it
            if c not in self.char_dict:
                self.char_dict[c] = 1
            else:
                self.char_dict[c] += 1








