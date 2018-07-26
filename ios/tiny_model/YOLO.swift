//
//  YOLO.swift
//  tiny_model
//
//  Created by wanwenhao on 2018/7/25.
//  Copyright © 2018年 wanwenhao. All rights reserved.
//

import Foundation
import UIKit
import CoreML

class YOLO {
    public static let inputWidth = 416
    public static let inputHeight = 416
    public static let maxBoundingBoxes = 20
    public static let shapePoint = 416 * 416 * 3
    // Tweak these values to get more or fewer predictions.
    let confidenceThreshold: Float = 0.2
    let iouThreshold: Float = 0.1
    
    struct Prediction {
        let classIndex: Int
        let score: Float
        let rect: CGRect
    }
    

    
    public init() { }
    
    public func predict(image: CVPixelBuffer) throws -> [Prediction] {
     
        if let output = try? model.prediction(image: image) {
            let comp = computeBoundingBoxes(features: output)
            return comp
        } else {
            return []
        }
        
    }
    
// when selected "tiny" coreML model , change target Membership
    let model = tiny()
    public func computeBoundingBoxes(features: tinyOutput) -> [Prediction] {

//  when selected "yolo" coreML model , change target Membership
//   let model = yolo()
//   public func computeBoundingBoxes(features: yoloOutput) -> [Prediction] {

    
        var predictions = [Prediction]()
 
        
        let boxesPerCell:Int = 3
        let numClasses:Int = labels.count
        let blockSize:Float = 416/13
        
        let imgChannel:Int = boxesPerCell*(numClasses+5)

        predictions.append(contentsOf:self.computeBoundingBoxes1(features: features.output1,
                                                                 blockSize: blockSize,
                                                                 boxesPerCell: boxesPerCell,
                                                                 numClasses: numClasses,
                                                                 imageChannel: imgChannel) )
        
        
        predictions.append(contentsOf:self.computeBoundingBoxes2(features: features.output2,
                                                                 blockSize: blockSize/2,
                                                                 boxesPerCell: boxesPerCell,
                                                                 numClasses: numClasses,
                                                                 imageChannel: imgChannel) )
        
// for yolo model
    
//        predictions.append(contentsOf:self.computeBoundingBoxes3(features: features.output3,
//                                                                 blockSize: blockSize/2,
//                                                                 boxesPerCell: boxesPerCell,
//                                                                 numClasses: numClasses,
//                                                                 imageChannel: imgChannel) )
        
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
        
    }
    
    
    public func computeBoundingBoxes1(features: MLMultiArray,blockSize:Float,boxesPerCell:Int ,numClasses:Int,imageChannel:Int) -> [Prediction] {
        
        
        let gridHeight:Int = Int(416/blockSize)
        let gridWidth:Int = Int(416/blockSize)
        
        assert(features.count == imageChannel * gridWidth * gridWidth)
        
        var predictions = [Prediction]()
   
        
        let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
        let channelStride = features.strides[0].intValue
        let yStride = features.strides[1].intValue
        let xStride = features.strides[2].intValue
        
        func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
            return channel*channelStride + y*yStride + x*xStride
        }
        
        for cy in 0..<gridHeight {
            for cx in 0..<gridWidth {
                for b in 0..<boxesPerCell {
                    
                    let channel = b*(numClasses + 5) //b = 0,1,2   numClasses = 4 boxesPerCell = 3
                    
                    // The fast way:
                    let tx = Float(featurePointer[offset(channel    , cx, cy)])
                    let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
                    let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
                    let th = Float(featurePointer[offset(channel + 3, cx, cy)])
                    let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
                    
                    let x = (Float(cx) + sigmoid(tx)) * blockSize  // blockSize = 32
                    let y = (Float(cy) + sigmoid(ty)) * blockSize
                    
                    
                    let w = exp(tw) * anchors1[2*b    ]
                    let h = exp(th) * anchors1[2*b + 1]
                    //                    print("1 tw:\(tw)  th:\(th)")
                    
                    let confidence = sigmoid(tc)
                    
                    var classes = [Float](repeating: 0, count: numClasses)
                    
                    
                    for c in 0..<numClasses {
                        classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
                    }
                    classes = softmax(classes)
                    
                    let (detectedClass, bestClassScore) = classes.argmax()
                    
                    let confidenceInClass = bestClassScore * confidence
                    
                    if confidenceInClass > confidenceThreshold {
                        let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                          width: CGFloat(w), height: CGFloat(h))
                        
                        let prediction = Prediction(classIndex: detectedClass,
                                                    score: confidenceInClass,
                                                    rect: rect)
                        predictions.append(prediction)
                    }
                }
            }
        }
        
        print("predictions1:\(predictions)")
        
        // We already filtered out any bounding boxes that have very low scores,
        // but there still may be boxes that overlap too much with others. We'll
        // use "non-maximum suppression" to prune those duplicate bounding boxes.
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    }
    
    public func computeBoundingBoxes2(features: MLMultiArray,blockSize:Float,boxesPerCell:Int,numClasses:Int,imageChannel:Int) -> [Prediction] {
        let gridHeight:Int = Int(416/blockSize)
        let gridWidth:Int = Int(416/blockSize)
        assert(features.count == imageChannel*gridWidth*gridHeight)
        
        var predictions = [Prediction]()
        
    
        
        let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
        let channelStride = features.strides[0].intValue
        let yStride = features.strides[1].intValue
        let xStride = features.strides[2].intValue
        
        func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
            return channel*channelStride + y*yStride + x*xStride
        }
        
        for cy in 0..<gridHeight {
            for cx in 0..<gridWidth {
                for b in 0..<boxesPerCell {
                    
                    let channel = b*(numClasses + 5) //b = 0,1,2   numClasses = 4 boxesPerCell = 3
                    
                    // The fast way:
                    let tx = Float(featurePointer[offset(channel    , cx, cy)])
                    let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
                    let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
                    let th = Float(featurePointer[offset(channel + 3, cx, cy)])
                    let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
                    
                    let x = (Float(cx) + sigmoid(tx)) * blockSize  // blockSize = 32
                    let y = (Float(cy) + sigmoid(ty)) * blockSize
                    
                    //                    print("2 tw:\(tw)  th:\(th)")
                    
                    //                    for i in 0..<imageChannel {
                    //                        print("i:\(i)  ,featurePointer:\(featurePointer[offset(i, cx, cy)])")
                    //                    }
                    let w = exp(tw) * anchors2[2*b    ]
                    let h = exp(th) * anchors2[2*b + 1]
                    
                    let confidence = sigmoid(tc)
                    
                    var classes = [Float](repeating: 0, count: numClasses)
                    
                    
                    for c in 0..<numClasses {
                        classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
                    }
                    classes = softmax(classes)
                    
                    let (detectedClass, bestClassScore) = classes.argmax()
                    
                    let confidenceInClass = bestClassScore * confidence
                    
                    if confidenceInClass > confidenceThreshold {
                        let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                          width: CGFloat(w), height: CGFloat(h))
                        
                        let prediction = Prediction(classIndex: detectedClass,
                                                    score: confidenceInClass,
                                                    rect: rect)
                        predictions.append(prediction)
                    }
                }
            }
        }
        
        print("predictions2:\(predictions)")
        
        // We already filtered out any bounding boxes that have very low scores,
        // but there still may be boxes that overlap too much with others. We'll
        // use "non-maximum suppression" to prune those duplicate bounding boxes.
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    }
   
    
    public func computeBoundingBoxes3(features: MLMultiArray,blockSize:Float,boxesPerCell:Int,numClasses:Int,imageChannel:Int) -> [Prediction] {
        let gridHeight:Int = Int(416/blockSize)
        let gridWidth:Int = Int(416/blockSize)
        assert(features.count == imageChannel*gridWidth*gridHeight)
        
        var predictions = [Prediction]()
        
        
        
        let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
        let channelStride = features.strides[0].intValue
        let yStride = features.strides[1].intValue
        let xStride = features.strides[2].intValue
        
        func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
            return channel*channelStride + y*yStride + x*xStride
        }
        
        for cy in 0..<gridHeight {
            for cx in 0..<gridWidth {
                for b in 0..<boxesPerCell {
                    
                    let channel = b*(numClasses + 5) //b = 0,1,2   numClasses = 4 boxesPerCell = 3
                    
                    // The fast way:
                    let tx = Float(featurePointer[offset(channel    , cx, cy)])
                    let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
                    let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
                    let th = Float(featurePointer[offset(channel + 3, cx, cy)])
                    let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
                    
                    let x = (Float(cx) + sigmoid(tx)) * blockSize  // blockSize = 32
                    let y = (Float(cy) + sigmoid(ty)) * blockSize
                    
                    //                    print("2 tw:\(tw)  th:\(th)")
                    
                    //                    for i in 0..<imageChannel {
                    //                        print("i:\(i)  ,featurePointer:\(featurePointer[offset(i, cx, cy)])")
                    //                    }
                    let w = exp(tw) * anchors2[2*b    ]
                    let h = exp(th) * anchors2[2*b + 1]
                    
                    let confidence = sigmoid(tc)
                    
                    var classes = [Float](repeating: 0, count: numClasses)
                    
                    
                    for c in 0..<numClasses {
                        classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
                    }
                    classes = softmax(classes)
                    
                    let (detectedClass, bestClassScore) = classes.argmax()
                    
                    let confidenceInClass = bestClassScore * confidence
                    
                    if confidenceInClass > confidenceThreshold {
                        let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                          width: CGFloat(w), height: CGFloat(h))
                        
                        let prediction = Prediction(classIndex: detectedClass,
                                                    score: confidenceInClass,
                                                    rect: rect)
                        predictions.append(prediction)
                    }
                }
            }
        }
        
        print("predictions3:\(predictions)")
        
        // We already filtered out any bounding boxes that have very low scores,
        // but there still may be boxes that overlap too much with others. We'll
        // use "non-maximum suppression" to prune those duplicate bounding boxes.
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    }
    
}

