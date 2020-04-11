"""This is a minimal implementation of a Table class 

Dear future developers:

    This should be entirely replaced by pandas.DataFrame .
"""
import collections


class Table:
    
    def __init__(self, column_names):
        
        self.column_names = column_names
        
        self.data = collections.OrderedDict(
            [(key, []) for key in column_names])
        
    def append(self, dict):
        
        for key in self.data.keys():
        
            if key in dict.keys():
            
                self.data[key].append(dict[key])
                
            else:
            
                self.data[key].append(None)
    
    def row(self, i):
    
        return tuple([val[i] for key, val in self.data.items()])
    
    def __len__(self):
    
        return len(self.data[self.column_names[0]])
    
    def __str__(self):
    
        return self.as_csv()
    
    def as_csv(self):
    
        def format_row(values):
        
            string = ""
            
            for val in values[:-1]:
            
                string += str(val) + ", "
                
            string += str(values[-1]) + "\n"
            
            return string
        
        string = ""
        
        string += format_row(self.column_names)
        
        for i in range(len(self)):
        
            string += format_row(self.row(i))
            
        return string
        
    def max(self, key):
    
        return max(list(filter(None.__ne__, self.data[key])))
   