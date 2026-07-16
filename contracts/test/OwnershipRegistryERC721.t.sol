// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {IERC721Errors} from "@openzeppelin/contracts/interfaces/draft-IERC6093.sol";
import {OwnershipRegistryERC721} from "../src/OwnershipRegistryERC721.sol";

contract OwnershipRegistryERC721Test is Test {
    OwnershipRegistryERC721 registry;

    address admin = makeAddr("admin");
    address creator = makeAddr("creator");
    address relayer = makeAddr("relayer");
    address stranger = makeAddr("stranger");

    bytes32 constant HASH_A = keccak256("artwork-a");
    bytes32 constant HASH_B = keccak256("artwork-b");

    function setUp() public {
        registry = new OwnershipRegistryERC721(admin);
    }

    // ---- register (self-service) ----

    function test_register_succeeds() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, true);

        assertEq(id, 1);
        assertEq(registry.ownerOf(id), creator);
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
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistryERC721.AlreadyRegistered.selector, HASH_A));
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
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistryERC721.NotRelayer.selector, relayer));
        registry.registerFor(creator, HASH_A, false);
    }

    function test_registerFor_succeedsForAuthorizedRelayer() public {
        vm.prank(admin);
        registry.setRelayer(relayer, true);

        vm.prank(relayer);
        uint256 id = registry.registerFor(creator, HASH_A, false);

        assertEq(registry.ownerOf(id), creator);
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

    // ---- standard ERC-721 transfer semantics (the actual point of the upgrade) ----

    function test_transferFrom_succeedsForOwner() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false);

        vm.prank(creator);
        registry.transferFrom(creator, stranger, id);

        assertEq(registry.ownerOf(id), stranger);
        (, address owner,,) = registry.verify(HASH_A);
        assertEq(owner, stranger);
    }

    function test_transferFrom_revertsForNonOwnerNonApproved() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false);

        vm.prank(stranger);
        vm.expectRevert(
            abi.encodeWithSelector(IERC721Errors.ERC721InsufficientApproval.selector, stranger, id)
        );
        registry.transferFrom(creator, stranger, id);
    }

    function test_transferFrom_revertsForUnknownId() public {
        vm.expectRevert(abi.encodeWithSelector(IERC721Errors.ERC721NonexistentToken.selector, 999));
        registry.transferFrom(creator, stranger, 999);
    }

    /// @dev The marketplace-compatibility gap OwnershipRegistry.sol's own
    ///      comment flagged: an approved operator (a marketplace contract,
    ///      in practice) can move the token without the owner submitting
    ///      the transaction themselves. The original contract has no
    ///      equivalent -- only the literal owner could call `transfer()`.
    function test_approvedOperator_canTransferOnOwnersBehalf() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false);

        vm.prank(creator);
        registry.approve(stranger, id);

        vm.prank(stranger);
        registry.transferFrom(creator, stranger, id);

        assertEq(registry.ownerOf(id), stranger);
    }

    function test_verify_reflectsOwnerAfterTransfer() public {
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false);
        vm.prank(creator);
        registry.transferFrom(creator, stranger, id);

        (, address owner,,) = registry.verify(HASH_A);
        assertEq(owner, stranger);
    }
}
