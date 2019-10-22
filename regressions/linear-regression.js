const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options){
        this.features = this.processFeatures(features)
        this.labels = tf.tensor(labels)
        this.mseHistory = []

        this.options = Object.assign({ 
            learningRate: 0.1, 
            iterations: 100,
            batchSize: 10
        }, options)
        this.weights = tf.zeros([this.features.shape[1],1])
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights)
        const differences = currentGuesses.sub(labels)
        const slopes = features.transpose().matMul(differences).div(features.shape[0])
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
    }

    // gradientDescent(){

    //     const currentGuesses = this.features.map(row => this.m * row[0] + this.b)

    //     const bSlope = (_.sum(
    //         currentGuesses.map((guess, i) => guess - this.labels[i][0])) 
    //         * 2) / this.features.length
    //     const mSlope = (_.sum(
    //         currentGuesses.map((guess, i) => -1 * this.features[i][0] * (this.labels[i][0] - guess))
    //     ) * 2) / this.features.length

    //     this.m = this.m - mSlope * this.options.learningRate
    //     this.b = this.b - bSlope * this.options.learningRate
    // }

    train(){
        const { batchSize, iterations } = this.options
        const { features, labels } = this
        const batchQuantity = Math.floor(features.shape[0] / batchSize)
        for (let i = 0; i < iterations; i++){
            for (let j = 0; j < batchQuantity; j++){
                const featureSlice = features.slice(
                    [j * batchSize, 0], 
                    [batchSize, -1])
                const labelSlice = labels.slice(
                    [j * batchSize, 0],
                    [batchSize, -1])
                this.gradientDescent(featureSlice, labelSlice)
            }
            this.recordMSE()
            this.updateLearningRate()
        }
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures)
        testLabels = tf.tensor(testLabels)

        const predictions = testFeatures.matMul(this.weights)

        const res = testLabels.sub(predictions).pow(2).sum().arraySync()
        const tot = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync()
        
        return 1 - res / tot

    }

    processFeatures(features) {
        features = tf.tensor(features)
        features = this.standardize(features)
        features = tf.ones([features.shape[0], 1]).concat(features, 1)

        return features
    }

    standardize(features) {
        if (!this.mean && !this.variance){
            const { mean, variance } = tf.moments(features, 0)
            this.mean = mean
            this.variance = variance
        }

        return features.sub(this.mean).div(this.variance.pow(0.5))
    }

    recordMSE(){
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .arraySync()
        this.mseHistory.unshift(mse)
    }

    updateLearningRate(){
        if (this.mseHistory.length < 2){
            return;
        }

        if (this.mseHistory[0] > this.mseHistory[1]){
            this.options.learningRate /= 2
        } else {
            this.options.learningRate *= 1.05
        }
    }
}

module.exports = LinearRegression