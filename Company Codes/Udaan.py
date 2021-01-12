# Final Result
class Theater:
    def __init__(self):
        self.house = {}

    def addScreen(self, screenName, details):
        if(screenName not in self.house):
            # Check if the Aisle seats exists in the seating arrangement - Also if given query is negative in nature
            for aisle in details[2:]:
                ai = int(aisle)
                if(ai <= 0):
                    return False
                
                ai -= 1 # 0th index
                row = ai//int(details[1])
                col = ai%int(details[1])
                
                if(row >= int(details[0]) or col >= int(details[1])):
                    return False
                    
            # Success -- 2d array with 0th index represent the actual seatings, 1st index represent Aisle seats in an array
            self.house[screenName] = [[[False for row in range(int(details[1]))] for cols in range(int(details[0]))], set(details[2:])]
            return True
        else:
            # Failure
            return False
    
    def reservation(self, screenName, details):
        
        if(screenName not in self.house):
            return False
        else:
            
            y = int(details[0])
            if(y <= 0 or y > len(self.house[screenName][0])):
                return False
            
            row = self.house[screenName][0][y-1]
            
            seatsReq = details[1:]
            for seat in seatsReq:
                seat = int(seat)
                if(seat <= 0 or row[seat-1] == True):
                    return False
            
            for seat in seatsReq:
                seat = int(seat)
                row[seat-1] = True
                
            self.house[screenName][0][y-1] = row
            return True
            
    def checkSeats(self, screenName, details):
        
        if(screenName not in self.house):
            return False
        else:
            detail = int(details[0])
            if(detail > len(self.house[screenName][0]) or detail <= 0):
                return False
            row = self.house[screenName][0][detail-1]
            res = []
            for i in range(0, len(row)):
                if(row[i] is False):
                    res.append(i+1)            
            return res 
    
    def suggest(self, screenName, details):
        
        if(screenName not in self.house):
            return False
        else:
            seats = int(details[0])
            y = int(details[1])
            choice = int(details[2])
            
            if(y > len(self.house[screenName][0]) or y <= 0):
                return False
            
            row = self.house[screenName][0][y-1]
            aisles = self.house[screenName][1] # all aisles we check these for contigous
            
            if(choice + seats <= len(row)):
                Flag = True
                
                # start one checked seperately -- maybe we need to check for end too seperately (loop mein choice +seats - 2 kar dena)
                if(row[choice - 1] is True):
                    Flag = False
                
                # check for aisles or reserved seats in between seats requested
                for i in range(choice, choice + seats - 1):
                    if(row[i] is True or str(i+1) in aisles): # if present index is an aisle
                        Flag = False
                        break
                    
                if(Flag):
                    res = []
                    for i in range(choice - 1, choice + seats - 1):
                        res.append(i+1)
                    return res 
            
            if(choice - seats >= 0):
                endWith = row[choice - seats : choice]
                Flag = True
                
                # end one checked
                if(row[choice-1] is True):
                    Flag = False
                
                # in between ones are checked
                for i in range(choice - seats, choice-1):
                    if(row[i] is True or str(i+1) in aisles):
                        Flag = False
                        break
               
                if(Flag):
                    res = []
                    for i in range(choice - seats, choice):
                        res.append(i+1)
                    return res 
            
            return ['none']

theater = Theater()

# Starter Code 
Queries = int(input())
while(Queries):
    
    query = input()
    
    string = query.split(' ')
    
    req = string[0]
    screen = string[1]
    details = string[2:]
    
    if(req == 'add-screen'):
        if(theater.addScreen(screen, details)):
            print('success')
        else:
            print('failure')
    elif(req == 'reserve-seat'):
        if(theater.reservation(screen, details)):
            print('success')
        else:
            print('failure')
    elif(req == 'get-unreserved-seats'):
        result = theater.checkSeats(screen, details)
        if(result is False):
            print('failure')
        else:
            for seat in result:
                print(seat, end = ' ')
            print('')
    elif(req == 'suggest-contiguous-seats'):
        result = theater.suggest(screen, details)
        if(result is not False):
            for res in result:
                print(res, end = ' ')
            print('')
    else:
        print('Some Kind of Error regarding request') # should not happen
    
    Queries -= 1


9
add-screen Screen1 12 10 4 5 8 9
add-screen Screen2 20 25 3 4 12 13 17 18
reserve-seat Screen1 4 5 6 7
reserve-seat Screen2 13 6 7 8 9 10
reserve-seat Screen2 13 4 5 6
get-unreserved-seats Screen2 13
suggest-contiguous-seats Screen1 3 3 4
suggest-contiguous-seats Screen2 4 12 4
suggest-contiguous-seats Screen2 4 10 3

7
add-screen Screen1 12 10 4 5 8 9
add-screen Screen2 20 25 3 4 12 13 17 18
reserve-seat Screen1 4 5 6 7
reserve-seat Screen2 13 6 7 8 9 10
reserve-seat Screen2 13 4 5 6
get-unreserved-seats Screen2 13
suggest-contiguous-seats Screen2 4 13 4