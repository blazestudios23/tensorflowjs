require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower','displacement','weight'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 10,
    batchSize: 10
})

regression.train()

// regression.weights.array().then(results => 
//     console.log("Updated M is: ", results[1][0], "Updated B is: ", results[0][0])
// )
const r2 = regression.test(testFeatures, testLabels)

plot({
    x: regression.mseHistory.reverse(),
    xLabel: "Iteration #",
    yLabel: "Mean Squared Error"
});




console.log("R2", r2);
