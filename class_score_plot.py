import matplotlib.pyplot as plt

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'): # If 'line' is not a header
                data.append([int(word) for word in line.split(',')])
    return data

if __name__ == '__main__':
    # Load score data
    class_kr = read_data('data/class_score_kr.csv')
    class_en = read_data('data/class_score_en.csv')

    # TODO) Prepare midterm, final, and total scores
    midterm_kr, final_kr = zip(*class_kr)
    total_kr = [40/125*midterm_kr + 60/100*final_kr for (midterm_kr, final_kr) in class_kr]
    midterm_en, final_en = zip(*class_en)
    total_en = [int(40/125*midterm_en + 60/100*final_en) for (midterm_en,final_en) in class_en]
   
    # TODO) Plot midterm/final scores as points
    plt.xlim(0,125)
    plt.ylim(0,100)
    plt.grid()
    plt.scatter(midterm_kr,final_kr,color='red',label='Korean')
    plt.scatter(midterm_en,final_en,color='blue',marker='+',label='English')
    plt.legend()
    plt.xlabel('Midterm scores')
    plt.ylabel('Final scores')
    plt.show()
    
    # TODO) Plot total scores as a histogram
    plt.xlim(0,100)
    standard=[num for num in range(0,101,5)]
    plt.hist(total_kr,color='red',label='korean',bins=standard,alpha=0.6)
    plt.hist(total_en,color='blue',label='English',bins=standard,alpha=0.6)
    plt.legend(loc='upper left')
    plt.xlabel('Total scores')
    plt.ylabel('The number of students')
    
    plt.show()