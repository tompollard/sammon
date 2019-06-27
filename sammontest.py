def main():

   import numpy as np
   from sklearn import datasets
   import matplotlib.pyplot as plt
   from sammon import sammon

   """Test sammon.py by plotting a projection of iris flower data. 
      Run sammontest() with no arguments.

   File        : sammontest.py
   Date        : 18 April 2014
   Author      : Tom J. Pollard (tom.pollard.11@ucl.ac.uk)

   Description : Script to test sammon.py by applying it
   				  to Fisher's iris dataset
                 http://en.wikipedia.org/wiki/Iris_flower_data_set

   Copyright   : (c) 2014, Tom J. Pollard

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
   (MIT License, http://www.opensource.org/licenses/mit-license.php)

   """
   # Load the iris data
   iris = datasets.load_iris()
   (x,index) = np.unique(iris.data,axis=0,return_index=True)
   target = iris.target[index]
   names = iris.target_names

   # Run the Sammon projection
   [y,E] = sammon(x, 2)

   # Plot
   plt.scatter(y[target ==0, 0], y[target ==0, 1], s=20, c='r', marker='o',label=names[0])
   plt.scatter(y[target ==1, 0], y[target ==1, 1], s=20, c='b', marker='D',label=names[1])
   plt.scatter(y[target ==2, 0], y[target ==2, 1], s=20, c='y', marker='v',label=names[2])
   plt.title('Sammon projection of iris flower data')
   plt.legend(loc=2)
   plt.show()

if __name__ == "__main__":
    main()

