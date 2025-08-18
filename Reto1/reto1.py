import random
import math
import time
import csv
import os

def createMatrix(rows, columns, minVal=0, maxVal=256):
    
    matrix = []
    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(random.randint(minVal, maxVal))
        matrix.append(row)
    return matrix

def processMatrix(matrix):
    
    init = time.time()
    
    min = 257
    max = -1
    sum = 0
    elements = len(matrix) * len(matrix[0])
    for fila in matrix:
        for element in fila:
            if element < min:
                min = element
            if element > max:
                max = element
            sum += element
           
    mean = sum / elements 
    standardDeviation = getStandardDeviation(matrix, mean)        
    totalTime = time.time() - init
    
    return min, max, mean, standardDeviation, totalTime

def getStandardDeviation(matrix, mean):
    sumDifference = 0
    for row in matrix:
        for element in row:
            difference = element - mean
            sumDifference += difference * difference
            
    elements = len(matrix) * len(matrix[0])
    variance = sumDifference / elements
    standardDeviation = math.sqrt(variance)
    
    return standardDeviation

def converToVector(matrix):
    vector = []
    for row in matrix:
        for element in row:
            vector.append(element)
            
    return vector

def saveCsv(vector):
    with open("vectorCsv.csv", 'w', newline='') as file:
        writter = csv.writer(file)
        
        for value in vector:
            writter.writerow([value])
            
def saveTextFile(vector):
    with open("vectorText.txt", 'w') as file:
        for i, value in enumerate(vector):
            if i == len(vector) - 1:
                file.write(str(value))
            else:
                file.write(str(value) + '\n')
                
def saveOutFile(vector):
        with open("vectorOut.out", 'w') as archivo:
            for i in range(0, len(vector), 50):
                line = vector[i:i+50]
                archivo.write(' '.join(map(str, line)) + '\n')
        
    
matrix = createMatrix(1000, 1000)
min, max, mean, standardDeviation, timeProcess = processMatrix(matrix)
vector = converToVector(matrix)

saveCsv(vector)
saveOutFile(vector)
saveTextFile(vector)

print(f"Minimum value: {min}")
print(f"Maximumimum value: {max}")
print(f"Mean value: {mean}")
print(f"Standard Deviation value: {standardDeviation}")
print(f"Process time: {timeProcess}")
print("Archivos guardados en:", os.getcwd())



    