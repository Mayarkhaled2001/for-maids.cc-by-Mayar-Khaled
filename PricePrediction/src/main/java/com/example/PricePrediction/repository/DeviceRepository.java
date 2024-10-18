package com.example.PricePrediction.repository;

import com.example.PricePrediction.entity.Device;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DeviceRepository extends JpaRepository<Device, Long> {
}

