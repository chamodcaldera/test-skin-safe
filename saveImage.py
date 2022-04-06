def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


def insertBLOB(file):

    binaryFile = convertToBinaryData(file)
    # file = convertToBinaryData(biodataFile)
    return  binaryFile
