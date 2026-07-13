// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {OwnershipRegistry} from "../src/OwnershipRegistry.sol";

/// @notice Deploy script for OwnershipRegistry.
/// Usage: forge script script/Deploy.s.sol --rpc-url amoy --broadcast --verify
contract Deploy is Script {
    function run() external returns (OwnershipRegistry registry) {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerKey);

        vm.startBroadcast(deployerKey);
        registry = new OwnershipRegistry(deployer);
        vm.stopBroadcast();
    }
}
