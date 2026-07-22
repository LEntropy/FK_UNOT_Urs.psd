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
    bytes32 constant NONCE = keccak256("nonce-1");

    function setUp() public {
        registry = new OwnershipRegistryERC721(admin);
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

    function _registerAsCreator(bytes32 contentHash, bool doNotTrain) private returns (uint256 id) {
        _commitAndAdvance(_selfCommitHash(contentHash, doNotTrain, creator, NONCE), creator);
        vm.prank(creator);
        id = registry.register(contentHash, doNotTrain, NONCE);
    }

    // ---- register (self-service) ----

    function test_register_succeeds() public {
        uint256 id = _registerAsCreator(HASH_A, true);

        assertEq(id, 1);
        assertEq(registry.ownerOf(id), creator);
        (bool exists, address owner, uint64 ts, bool doNotTrain) = registry.verify(HASH_A);
        assertTrue(exists);
        assertEq(owner, creator);
        assertEq(ts, uint64(block.timestamp));
        assertTrue(doNotTrain);
    }

    function test_register_incrementsIdAcrossHashes() public {
        uint256 idA = _registerAsCreator(HASH_A, false);
        uint256 idB = _registerAsCreator(HASH_B, false);

        assertEq(idA, 1);
        assertEq(idB, 2);
    }

    function test_register_revertsOnDuplicateHash() public {
        _registerAsCreator(HASH_A, false);

        bytes32 strangerNonce = keccak256("nonce-2");
        _commitAndAdvance(_selfCommitHash(HASH_A, false, stranger, strangerNonce), stranger);
        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistryERC721.AlreadyRegistered.selector, HASH_A));
        registry.register(HASH_A, false, strangerNonce);
    }

    // ---- commit-reveal guard ----

    function test_register_revertsWithoutAPriorCommit() public {
        vm.prank(creator);
        bytes32 commitHash = _selfCommitHash(HASH_A, false, creator, NONCE);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistryERC721.CommitNotFound.selector, commitHash));
        registry.register(HASH_A, false, NONCE);
    }

    function test_register_revertsWhenRevealedBeforeMinCommitAge() public {
        vm.prank(creator);
        registry.commit(_selfCommitHash(HASH_A, false, creator, NONCE));

        vm.prank(creator);
        vm.expectRevert();
        registry.register(HASH_A, false, NONCE);
    }

    function test_commitReveal_defeatsACopyCatFrontRunner() public {
        _commitAndAdvance(_selfCommitHash(HASH_A, false, creator, NONCE), creator);

        bytes32 attackerNonce = keccak256("attacker-nonce");
        vm.prank(stranger);
        registry.commit(_selfCommitHash(HASH_A, false, stranger, attackerNonce));
        vm.prank(stranger);
        vm.expectRevert();
        registry.register(HASH_A, false, attackerNonce);

        vm.prank(creator);
        uint256 id = registry.register(HASH_A, false, NONCE);
        assertEq(id, 1);
        assertEq(registry.ownerOf(id), creator);

        vm.roll(block.number + registry.MIN_COMMIT_AGE());
        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistryERC721.AlreadyRegistered.selector, HASH_A));
        registry.register(HASH_A, false, attackerNonce);
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
        vm.expectRevert(abi.encodeWithSelector(OwnershipRegistryERC721.NotRelayer.selector, relayer));
        registry.registerFor(creator, HASH_A, false, NONCE);
    }

    function test_registerFor_succeedsForAuthorizedRelayer() public {
        vm.prank(admin);
        registry.setRelayer(relayer, true);

        _commitAndAdvance(_relayerCommitHash(creator, HASH_A, false, relayer, NONCE), relayer);

        vm.prank(relayer);
        uint256 id = registry.registerFor(creator, HASH_A, false, NONCE);

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
        uint256 id = _registerAsCreator(HASH_A, false);

        vm.prank(creator);
        registry.transferFrom(creator, stranger, id);

        assertEq(registry.ownerOf(id), stranger);
        (, address owner,,) = registry.verify(HASH_A);
        assertEq(owner, stranger);
    }

    function test_transferFrom_revertsForNonOwnerNonApproved() public {
        uint256 id = _registerAsCreator(HASH_A, false);

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
        uint256 id = _registerAsCreator(HASH_A, false);

        vm.prank(creator);
        registry.approve(stranger, id);

        vm.prank(stranger);
        registry.transferFrom(creator, stranger, id);

        assertEq(registry.ownerOf(id), stranger);
    }

    function test_verify_reflectsOwnerAfterTransfer() public {
        uint256 id = _registerAsCreator(HASH_A, false);
        vm.prank(creator);
        registry.transferFrom(creator, stranger, id);

        (, address owner,,) = registry.verify(HASH_A);
        assertEq(owner, stranger);
    }

    // ---- tokenURI / metadata ----

    function test_tokenURI_isEmptyUntilBaseURIIsSet() public {
        uint256 id = _registerAsCreator(HASH_A, false);
        assertEq(registry.tokenURI(id), "");
    }

    function test_tokenURI_reflectsConfiguredBaseURI() public {
        uint256 id = _registerAsCreator(HASH_A, false);

        vm.prank(admin);
        registry.setBaseURI("https://api.dontai.app/nft-metadata/");

        assertEq(registry.tokenURI(id), "https://api.dontai.app/nft-metadata/1");
    }

    function test_setBaseURI_revertsForNonAdmin() public {
        vm.prank(stranger);
        vm.expectRevert();
        registry.setBaseURI("https://evil.example/");
    }

    // ---- pause ----

    function test_pause_blocksCommitAndRegisterButNotTransfers() public {
        uint256 id = _registerAsCreator(HASH_A, false);

        vm.prank(admin);
        registry.pause();

        vm.prank(creator);
        vm.expectRevert();
        registry.commit(_selfCommitHash(HASH_B, false, creator, NONCE));

        // Standard ERC721 transfers deliberately stay live while paused --
        // see pause()'s own doc comment on why marketplace trading of
        // already-minted tokens isn't frozen by this switch.
        vm.prank(creator);
        registry.transferFrom(creator, stranger, id);
        assertEq(registry.ownerOf(id), stranger);
    }

    function test_pause_revertsForNonAdmin() public {
        vm.prank(stranger);
        vm.expectRevert();
        registry.pause();
    }

    function test_unpause_restoresRegistration() public {
        vm.prank(admin);
        registry.pause();
        vm.prank(admin);
        registry.unpause();

        uint256 id = _registerAsCreator(HASH_A, false);
        assertEq(id, 1);
    }

    // ---- Ownable2Step ----

    function test_ownershipTransfer_requiresAcceptanceByNewOwner() public {
        vm.prank(admin);
        registry.transferOwnership(stranger);

        assertEq(registry.owner(), admin);
        assertEq(registry.pendingOwner(), stranger);

        vm.prank(stranger);
        registry.acceptOwnership();

        assertEq(registry.owner(), stranger);
        vm.prank(admin);
        vm.expectRevert();
        registry.setRelayer(relayer, true);
    }
}
