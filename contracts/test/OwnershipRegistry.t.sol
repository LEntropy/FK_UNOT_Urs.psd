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
    bytes32 constant NONCE = keccak256("nonce-1");

    function setUp() public {
        registry = new OwnershipRegistry(admin);
    }

    // ---- helpers ----

    function _commitAndAdvance(bytes32 commitHash, address committer) private {
        vm.prank(committer);
        registry.commit(commitHash);
        vm.roll(block.number + registry.MIN_COMMIT_AGE());
    }

    function _selfCommitHash(bytes32 contentHash, bool doNotTrain, address caller, bytes32 nonce)
        private
        pure
        returns (bytes32)
    {
        return keccak256(abi.encode(contentHash, doNotTrain, caller, nonce));
    }

    function _relayerCommitHash(address owner, bytes32 contentHash, bool doNotTrain, address relayerAddr, bytes32 nonce)
        private
        pure
        returns (bytes32)
    {
        return keccak256(abi.encode(contentHash, doNotTrain, owner, relayerAddr, nonce));
    }

    // ---- register (self-service) ----

    function test_register_succeeds() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, true, creator, NONCE), creator);

        vm.prank(creator);
        uint256 id = registry.register(HASH_A, true, NONCE);

        assertEq(id, 1);
        (bool exists, address owner, uint64 ts, bool doNotTrain) = registry.verify(HASH_A);
        assertTrue(exists);
        assertEq(owner, creator);
        assertEq(ts, uint64(block.timestamp));
        assertTrue(doNotTrain);
    }

    function test_register_incrementsIdAcrossHashes() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        uint256 idA = registry.register(HASH_A, false, NONCE);

        _commitAndAdvance(_selfCommitHash(HASH_B, false, creator, NONCE), creator);
        vm.prank(creator);
        uint256 idB = registry.register(HASH_B, false, NONCE);

        assertEq(idA, 1);
        assertEq(idB, 2);
    }

    function test_register_revertsOnDuplicateHash() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        registry.register(HASH_A, false, NONCE);

        bytes32 strangerNonce = keccak256("nonce-2");
        _commitAndAdvance(_selfCommitHash(HASH_A, false, stranger, strangerNonce), stranger);
        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.AlreadyRegistered.selector, HASH_A));
        registry.register(HASH_A, false, strangerNonce);
    }

    // ---- commit-reveal guard ----

    function test_register_revertsWithoutAPriorCommit() public {
        vm.prank(creator);
        bytes32 commitHash = _selfCommitHash(HASH_A, false, creator, NONCE);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.CommitNotFound.selector, commitHash));
        registry.register(HASH_A, false, NONCE);
    }

    function test_register_revertsWhenRevealedBeforeMinCommitAge() public {
        vm.prank(creator);
        registry.commit(_selfCommitHash(HASH_A, false, creator, NONCE));
        // No vm.roll -- still the same block the commit was made in.

        vm.prank(creator);
        vm.expectRevert(); // CommitTooRecent (exact readyAtBlock not asserted, just that it reverts)
        registry.register(HASH_A, false, NONCE);
    }

    function test_register_revertsWhenNonceDoesNotMatchTheCommit() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);

        vm.prank(creator);
        bytes32 wrongNonce = keccak256("wrong-nonce");
        vm.expectRevert(
            abi.encodeWithSelector(
                OwnershipRegistry.CommitNotFound.selector, _selfCommitHash(HASH_A, false, creator, wrongNonce)
            )
        );
        registry.register(HASH_A, false, wrongNonce);
    }

    function test_commitReveal_defeatsACopyCatFrontRunner() public {
        // The real registrant commits first and waits out MIN_COMMIT_AGE.
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);

        // An attacker who somehow learned HASH_A (e.g. by watching the
        // creator's reveal transaction sit in the mempool) tries to race
        // it by committing + revealing their own claim to the same hash in
        // the very next block -- the whole point of MIN_COMMIT_AGE is that
        // this can never land before the legitimate reveal that already
        // finished waiting.
        bytes32 attackerNonce = keccak256("attacker-nonce");
        vm.prank(stranger);
        registry.commit(_selfCommitHash(HASH_A, false, stranger, attackerNonce));
        // Attacker's commit isn't old enough yet -- reveal must fail even
        // though they now also know HASH_A.
        vm.prank(stranger);
        vm.expectRevert();
        registry.register(HASH_A, false, attackerNonce);

        // Legitimate registrant reveals first, successfully.
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false, NONCE);
        assertEq(id, 1);

        // Attacker's reveal, even once their own commit matures, now hits
        // AlreadyRegistered instead of stealing the slot.
        vm.roll(block.number + registry.MIN_COMMIT_AGE());
        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.AlreadyRegistered.selector, HASH_A));
        registry.register(HASH_A, false, attackerNonce);
    }

    function test_commit_isOneTimeUse() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        registry.register(HASH_A, false, NONCE);

        // Re-registering a different hash by replaying the exact same
        // commitHash (which was deleted on first use) must fail, not
        // silently succeed off leftover state.
        vm.prank(creator);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.CommitNotFound.selector, _selfCommitHash(HASH_A, false, creator, NONCE)));
        registry.register(HASH_A, false, NONCE);
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
        _commitAndAdvance(_relayerCommitHash(creator, HASH_A, false, relayer, NONCE), relayer);

        vm.prank(relayer);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.NotRelayer.selector, relayer));
        registry.registerFor(creator, HASH_A, false, NONCE);
    }

    function test_registerFor_succeedsForAuthorizedRelayer() public {
        vm.prank(admin);
        registry.setRelayer(relayer, true);

        _commitAndAdvance(_relayerCommitHash(creator, HASH_A, false, relayer, NONCE), relayer);

        vm.prank(relayer);
        uint256 id = registry.registerFor(creator, HASH_A, false, NONCE);

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
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false, NONCE);

        vm.prank(creator);
        registry.transfer(id, stranger);

        (, address owner,,) = registry.verify(HASH_A);
        assertEq(owner, stranger);
    }

    function test_transfer_revertsForNonOwner() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false, NONCE);

        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.NotOwner.selector, id, stranger));
        registry.transfer(id, stranger);
    }

    function test_transfer_revertsForUnknownId() public {
        vm.prank(creator);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistry.NotFound.selector, 999));
        registry.transfer(999, stranger);
    }

    // ---- pause ----

    function test_pause_blocksCommitRegisterAndTransfer() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false, NONCE);

        vm.prank(admin);
        registry.pause();

        vm.prank(creator);
        vm.expectRevert();
        registry.commit(_selfCommitHash(HASH_B, false, creator, NONCE));

        vm.prank(creator);
        vm.expectRevert();
        registry.transfer(id, stranger);
    }

    function test_pause_revertsForNonAdmin() public {
        vm.prank(stranger);
        vm.expectRevert();
        registry.pause();
    }

    function test_unpause_restoresNormalOperation() public {
        vm.prank(admin);
        registry.pause();
        vm.prank(admin);
        registry.unpause();

        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false, NONCE);
        assertEq(id, 1);
    }

    function test_verify_stillWorksWhilePaused() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);
        vm.prank(creator);
        registry.register(HASH_A, false, NONCE);

        vm.prank(admin);
        registry.pause();

        (bool exists,,,) = registry.verify(HASH_A);
        assertTrue(exists);
    }

    // ---- Ownable2Step ----

    function test_ownershipTransfer_requiresAcceptanceByNewOwner() public {
        vm.prank(admin);
        registry.transferOwnership(stranger);

        // Old owner loses admin rights immediately on the *first* step
        // already being underway is not true for Ownable2Step -- admin
        // stays owner until stranger explicitly accepts.
        assertEq(registry.owner(), admin);
        assertEq(registry.pendingOwner(), stranger);

        vm.prank(stranger);
        registry.acceptOwnership();

        assertEq(registry.owner(), stranger);
        vm.prank(admin);
        vm.expectRevert();
        registry.setRelayer(relayer, true); // old owner can no longer administer
    }

    function test_ownershipTransfer_mistypedAddressDoesNotBrickAdmin() public {
        address typo = makeAddr("typo-victim");
        vm.prank(admin);
        registry.transferOwnership(typo);

        // Unlike single-step Ownable, a typo here is recoverable: admin is
        // still the real owner and can just start over with the right address.
        assertEq(registry.owner(), admin);
        vm.prank(admin);
        registry.transferOwnership(stranger);
        vm.prank(stranger);
        registry.acceptOwnership();
        assertEq(registry.owner(), stranger);
    }
}
