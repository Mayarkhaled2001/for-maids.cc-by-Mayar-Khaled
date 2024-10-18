package com.example.PricePrediction.controller;

import com.example.PricePrediction.entity.Device;
import com.example.PricePrediction.repository.DeviceRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/devices")
public class DeviceController {

    @Autowired
    private DeviceRepository deviceRepository;

    private final String PYTHON_API_URL = "http://localhost:6000/predict";

    // GET /api/devices/ - Retrieve all devices
    @GetMapping("/")
    public List<Device> getAllDevices() {
        return deviceRepository.findAll();
    }

    // GET /api/devices/{id} - Retrieve a specific device by ID
    @GetMapping("/{id}")
    public Optional<Device> getDeviceById(@PathVariable Long id) {
        return deviceRepository.findById(id);
    }

    // POST /api/devices - Add a new device
    @PostMapping("/")
    public Device addDevice(@RequestBody Device device) {
        return deviceRepository.save(device);
    }

    // POST /api/predict/{deviceId} - Predict price and save it in the database
    @PostMapping("/predict/{deviceId}")
    @Transactional
    public Device predictAndSavePrice(@PathVariable Long deviceId) {
        Device device = deviceRepository.findById(deviceId)
                .orElseThrow(() -> new RuntimeException("Device not found"));

        // Call Python API to predict price
        RestTemplate restTemplate = new RestTemplate();
        String predictedPrice = restTemplate.postForObject(PYTHON_API_URL, device.getSpecs(), String.class);

        // Save the prediction in the device entity
        device.setPredictedPrice(predictedPrice);
        return deviceRepository.save(device);
    }
}
