package com.example.PricePrediction.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Device {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String specs;
    private String predictedPrice;

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getSpecs() { return specs; }
    public void setSpecs(String specs) { this.specs = specs; }

    public String getPredictedPrice() { return predictedPrice; }
    public void setPredictedPrice(String predictedPrice) { this.predictedPrice = predictedPrice; }
}

