// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {OwnershipRegistry} from "../src/OwnershipRegistry.sol";

contract OwnershipRegistryTest is Test {
    OwnershipRegistry registry;

    address admin = makeAddr("admin");
    address creator = makeAddr("creator");
    address relayer = makeAddr("relayer");
    address stranger = makeAddr("stranger");

    bytes32 constant HASH_A = keccak256("artwork-a");
    bytes32 constant HASH_B = keccak256("artwork-b");

    function setUp() public {
        registry = new OwnershipRegistry(admin);
    }

    // ---- register (self-service) ----

    function test_register_succeeds() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, true);

        assertEq(id, 1);
        (bool exists, address owner, uint64 ts, bool doNotTrain) = registry.verify(HASH_A);
        assertTrue(exists);
        assertEq(owner, creator);
        assertEq(ts, uint64(block.timestamp));
        assertTrue(doNotTrain);
    }

    function test_register_incrementsIdAcrossHashes() public {
        vm.prank(creator);
        uint256 idA = registry.register(HASH_A, false);
        vm.prank(creator);
        uint256 idB = registry.register(HASH_B, false);

        assertEq(idA, 1);
        assertEq(idB, 2);
    }

    function test_register_revertsOnDuplicateHash() public {
        vm.prank(creator);
        registry.register(HASH_A, false);

        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.AlreadyRegistered.selector, HASH_A));
        registry.register(HASH_A, false);
    }

    // ---- verify ----

    function test_verify_returnsNotFoundForUnknownHash() public view {
        (bool exists, address owner, uint64 ts, bool doNotTrain) = registry.verify(HASH_A);
        assertFalse(exists);
        assertEq(owner, address(0));
        assertEq(ts, 0);
        assertFalse(doNotTrain);
    }

    // ---- relayer-sponsored registration ----

    function test_registerFor_revertsForUnauthorizedRelayer() public {
        vm.prank(relayer);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.NotRelayer.selector, relayer));
        registry.registerFor(creator, HASH_A, false);
    }

    function test_registerFor_succeedsForAuthorizedRelayer() public {
        vm.prank(admin);
        registry.setRelayer(relayer, true);

        vm.prank(relayer);
        uint256 id = registry.registerFor(creator, HASH_A, false);

        (bool exists, address owner,,) = registry.verify(HASH_A);
        assertTrue(exists);
        assertEq(owner, creator);
        assertEq(id, 1);
    }

    function test_setRelayer_revertsForNonAdmin() public {
        vm.prank(stranger);
        vm.expectRevert();
        registry.setRelayer(relayer, true);
    }

    // ---- transfer ----

    function test_transfer_succeedsForOwner() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false);

        vm.prank(creator);
        registry.transfer(id, stranger);

        (, address owner,,) = registry.verify(HASH_A);
        assertEq(owner, stranger);
    }

    function test_transfer_revertsForNonOwner() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false);

        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.NotOwner.selector, id, stranger));
        registry.transfer(id, stranger);
    }

    function test_transfer_revertsForUnknownId() public {
        vm.prank(creator);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.NotFound.selector, 999));
        registry.transfer(999, stranger);
    }
}
