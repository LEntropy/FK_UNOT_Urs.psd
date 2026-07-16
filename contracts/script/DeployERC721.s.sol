// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {OwnershipRegistryERC721} from "../src/OwnershipRegistryERC721.sol";

/// @notice Deploy script for OwnershipRegistryERC721 -- see README.md's
///         "ERC-721 migration" section before running this against Amoy or
///         mainnet. Not run as part of any CI/deploy pipeline; this is
///         prepared, not scheduled.
/// Usage: forge script script/DeployERC721.s.sol --rpc-url amoy --broadcast --verify
contract DeployERC721 is Script {
    function run() external returns (OwnershipRegistryERC721 registry) {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerKey);

        vm.startBroadcast(deployerKey);
        registry = new OwnershipRegistryERC721(deployer);
        vm.stopBroadcast();
    }
}
